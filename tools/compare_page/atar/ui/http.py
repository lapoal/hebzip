"""A tiny WSGI layer: request, response, router, and the root safety net. Standard library only.

The body limit is a config row passed to :class:`App`
(``limits.max_upload_bytes``); a :class:`Busy` error answers 409; the static folder sends a
strict CSP (nothing may load from outside this computer); the traceback detail of a 500 is
not sent to the page (it could quote text) - the type name only.

Why not Flask/FastAPI: the UI must merge into the main project by copying
files and run offline on one personal computer. A framework would be one more thing to install,
pin and keep in step with mypy-strict; what the UI needs from one is ~150 lines — JSON in,
JSON out, a path pattern, static files.

Always a valid answer, with a safety net at the root: every handler runs inside
:func:`App.__call__`'s net. A typed atar error becomes a 4xx with its Hebrew message; any
other exception becomes a 500 with a Hebrew sentence, the error type name, and a
LOUD log line. The browser never gets an empty page.
"""

from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, unquote

from atar.core.errors import AtarError, Busy
from atar.core.logging import get_logger

_log = get_logger("web")
CSP: Final = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
              "connect-src 'self'; font-src 'none'; object-src 'none'; base-uri 'none'; "
              "form-action 'none'; frame-ancestors 'none'")
StartResponse = Callable[[str, list[tuple[str, str]]], object]
Environ = Mapping[str, Any]


class HttpError(Exception):
    """A deliberate HTTP answer with a Hebrew message (404, 409 ...)."""

    def __init__(self, status: int, message: str, code: str = "", extra: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.extra = extra


@dataclass(frozen=True, slots=True)
class Request:
    """One request, already decoded."""

    method: str
    path: str
    query: Mapping[str, str]
    body: bytes
    params: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        """The JSON body as an object (empty body = empty object)."""
        if not self.body:
            return {}
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise HttpError(400, "הבקשה אינה JSON תקין") from error
        if not isinstance(data, dict):
            raise HttpError(400, "הבקשה צריכה להיות אובייקט JSON")
        return data


@dataclass(frozen=True, slots=True)
class Response:
    """Status, headers, body bytes."""

    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: tuple[tuple[str, str], ...] = ()


def json_response(data: Any, status: int = 200) -> Response:
    """Serialize to UTF-8 JSON (Hebrew kept readable)."""
    return Response(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))


def file_response(path: Path, download_name: str | None = None) -> Response:
    """A file from disk; ``download_name`` makes the browser save it."""
    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if kind.startswith("text/") or kind.endswith(("javascript", "json")):
        kind += "; charset=utf-8"
    headers: tuple[tuple[str, str], ...] = ()
    if download_name:
        quoted = re.sub(r"[^\w.\-]", "_", download_name.encode("ascii", "replace").decode())
        headers += (("Content-Disposition",
                     f"attachment; filename=\"{quoted}\"; filename*=UTF-8''"
                     + _rfc5987(download_name)),)
    return Response(200, path.read_bytes(), kind, headers)


def _rfc5987(name: str) -> str:
    return "".join(c if c.isascii() and (c.isalnum() or c in "-._") else
                   "".join(f"%{b:02X}" for b in c.encode("utf-8")) for c in name)


Handler = Callable[[Request], Response]
_STATUS: Final = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
                  409: "Conflict", 413: "Payload Too Large", 422: "Unprocessable Entity",
                  500: "Internal Server Error"}


class Router:
    """``/api/books/{book}/jobs`` style patterns; one handler per (method, pattern)."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, re.Pattern[str], Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        """Register a handler; ``{name}`` matches one path part."""
        regex = "^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$"
        self.routes.append((method, re.compile(regex), handler))

    def match(self, method: str, path: str) -> tuple[Handler, dict[str, str]]:
        """The handler and its path parameters, or a 404/405."""
        allowed = False
        for verb, regex, handler in self.routes:
            found = regex.match(path)
            if found is None:
                continue
            if verb == method:
                return handler, {k: unquote(v) for k, v in found.groupdict().items()}
            allowed = True
        if allowed:
            raise HttpError(405, "פעולה לא נתמכת בכתובת הזאת")
        raise HttpError(404, "הכתובת לא נמצאה")


def _path(environ: Environ) -> str:
    raw = str(environ.get("PATH_INFO", "/"))
    try:  # WSGI hands the path as latin-1 decoded bytes (PEP 3333)
        return raw.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def read_request(environ: Environ, max_body: int) -> Request:
    """Decode method, path, query and body (bounded)."""
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length > max_body:
        raise HttpError(413, f"הקובץ גדול מדי (מעל {max_body // (1024 * 1024)} MB)")
    body = environ["wsgi.input"].read(length) if length > 0 else b""
    query = {k: v[0] for k, v in parse_qs(str(environ.get("QUERY_STRING", ""))).items()}
    return Request(str(environ.get("REQUEST_METHOD", "GET")).upper(), _path(environ),
                   query, body)


def error_body(message: str, detail: str = "", code: str = "", extra: Any = None) -> Response:
    """The one error shape the browser knows: Hebrew message + collapsible detail."""
    return json_response({"error": message, "detail": detail, "code": code, "extra": extra})


class App:
    """The WSGI callable: API routes, static files, and the safety net."""

    def __init__(self, router: Router, static: Path, max_body: int) -> None:
        self.router = router
        self.static = static
        self.max_body = max_body

    def __call__(self, environ: Environ, start: StartResponse) -> Iterable[bytes]:
        response = self.respond(environ)
        headers = [("Content-Type", response.content_type),
                   ("Content-Length", str(len(response.body))),
                   ("X-Content-Type-Options", "nosniff"), ("Content-Security-Policy", CSP),
                   ("Referrer-Policy", "no-referrer"), ("Cache-Control", "no-store"),
                   *response.headers]
        start(f"{response.status} {_STATUS.get(response.status, 'Error')}", headers)
        return [response.body]

    def respond(self, environ: Environ) -> Response:
        """Route one request; never raises."""
        path = _path(environ)
        try:
            request = read_request(environ, self.max_body)
            if not path.startswith("/api/"):
                return self.static_file(path)
            handler, params = self.router.match(request.method, path)
            return handler(Request(request.method, path, request.query, request.body, params))
        except HttpError as error:
            return _with_status(error_body(str(error), code=error.code, extra=error.extra),
                                error.status)
        except Busy as error:
            return _with_status(error_body(str(error), "Busy"), 409)
        except AtarError as error:
            _log.warning("request failed", error=type(error).__name__)
            return _with_status(error_body(str(error), type(error).__name__), 422)
        except Exception as error:  # noqa: BLE001 - root safety net (last-resort)
            _log.error("unexpected failure", error=type(error).__name__)
            return _with_status(error_body("תקלה לא צפויה. הפעולה לא בוצעה — אפשר לנסות שוב.",
                                           type(error).__name__), 500)

    def static_file(self, path: str) -> Response:
        """``/`` → index.html; anything else must be a file inside the static folder."""
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (self.static / name).resolve()
        if self.static.resolve() not in target.parents or not target.is_file():
            raise HttpError(404, "הקובץ לא נמצא")
        return file_response(target)


def _with_status(response: Response, status: int) -> Response:
    return Response(status, response.body, response.content_type, response.headers)
