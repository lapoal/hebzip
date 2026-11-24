"""Routes: ping, status (poll), and upload. Thin - converts request <-> dict, decides nothing.

Upload shape: ``POST /api/jobs?name=<file name>`` with the raw file bytes as the body (no
multipart parsing, no base64 - fewer moving parts). The name is used only to pick txt/docx
and to show on the page; it is never logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atar.config.load import ROOT, Settings
from atar.services.jobs import JobBoard
from atar.ui.http import App, HttpError, Request, Response, Router, json_response

APP_NAME = "hebzip-atar"
STATIC = ROOT / "static"


def build_app(settings: Settings, board: JobBoard, static: Path = STATIC) -> App:
    """Wire the three routes to one job board."""
    versions = board.versions()
    router = Router()

    def ping(request: Request) -> Response:
        del request
        return json_response({"app": APP_NAME})

    def status(request: Request) -> Response:
        del request
        payload: dict[str, Any] = {"job": board.snapshot(), "versions": versions,
                                   "services": board.services(),
                                   "poll_ms": settings.poll_ms,
                                   "max_text_bytes": settings.max_text_bytes,
                                   "max_upload_bytes": settings.max_upload_bytes}
        return json_response(payload)

    def upload(request: Request) -> Response:
        name = request.query.get("name", "").strip()
        if not name:
            raise HttpError(400, "חסר שם הקובץ")
        if not request.body:
            raise HttpError(400, "הקובץ ריק")
        number = board.start(name, request.body)
        return json_response({"job": number}, 202)

    router.add("GET", "/api/ping", ping)
    router.add("GET", "/api/status", status)
    router.add("POST", "/api/jobs", upload)
    return App(router, static, settings.max_upload_bytes)
