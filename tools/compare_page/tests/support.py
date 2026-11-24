"""Test helpers: synthetic Hebrew text, a .docx builder, a real server on a free port.

Privacy: tests read no real user text.
The Hebrew here is written for the tests. Every test writes only under ``tests/_work`` and
that folder is emptied after each test (conftest) and checked empty at the end of the suite.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from wsgiref.simple_server import make_server

from atar.config.load import ROOT, Settings, ToolRow, load
from atar.services.jobs import JobBoard
from atar.ui.app import build_app
from atar.ui.server import QuietHandler, ThreadingServer, make_board

WORK = ROOT / "tests" / "_work"
HEBREW = ("שלום לכולם. היום נלמד על אופניים ועל הדרך הנכונה לרכוב בעיר.\n"
          "האופניים הם כלי תחבורה פשוט, זול ושקט, והם טובים לבריאות.\n"
          "בעיר גדולה כדאי לרכוב בשביל המיועד לאופניים, ולחבוש קסדה תמיד.\n")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx_bytes(body_xml: str) -> bytes:
    """A minimal valid .docx whose body is ``body_xml`` (namespace prefix ``w``)."""
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:document xmlns:w="{W_NS}"><w:body>{body_xml}</w:body></w:document>')
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns='
                   '"http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", document.encode("utf-8"))
    return out.getvalue()


def paragraph(text: str, with_tab_stops: bool = True) -> str:
    """One w:p with tab-stop properties (the <w:tabs> that fooled the old regex)."""
    props = ('<w:pPr><w:bidi/><w:tabs><w:tab w:val="right" w:pos="9026"/></w:tabs></w:pPr>'
             if with_tab_stops else "")
    return f'{props}<w:r><w:rPr><w:rtl/></w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'


def settings(tools: tuple[ToolRow, ...] | None = None, max_text_bytes: int | None = None) -> Settings:
    """The real config, with material/logs moved under tests/_work and optional rows."""
    base = load()
    s = replace(base, material=WORK / "material", logs=WORK / "logs")
    if tools is not None:
        s = replace(s, tools=tools)
    return replace(s, max_text_bytes=max_text_bytes) if max_text_bytes is not None else s


def row(tool_id: str, kind: str, options: dict[str, Any], ours: bool = False,
        timeout_s: float = 60.0) -> ToolRow:
    return ToolRow(tool_id, tool_id, kind, ours, timeout_s, options)


FAST = (row("zip", "gzip", {"level": 9}), row("bz2", "bz2", {"level": 9}),
        row("xz", "xz", {"preset": 9, "extreme": True}),
        row("ppmd", "ppmd7", {"order": 8, "memory_mb": 64}))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def live_server(s: Settings, board: JobBoard | None = None) -> Iterator[tuple[str, JobBoard]]:
    """The real WSGI app on a real socket (the path the browser takes)."""
    board = board or make_board(s)
    server = make_server("127.0.0.1", free_port(), build_app(s, board),
                         server_class=ThreadingServer, handler_class=QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    board.pool.start_all()  # like the real server: after the bind, in the background
    try:
        yield f"http://127.0.0.1:{server.server_port}", board
    finally:
        board.wait(600)
        server.shutdown()
        server.server_close()
        board.pool.stop_all()


def call(url: str, data: bytes | None = None) -> tuple[int, dict[str, Any]]:
    """GET (or POST raw bytes) and decode the JSON answer, also for 4xx."""
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def upload(base: str, name: str, data: bytes) -> tuple[int, dict[str, Any]]:
    return call(f"{base}/api/jobs?name={urllib.parse.quote(name)}", data)


def wait_done(base: str, budget_s: float = 600) -> dict[str, Any]:
    """Poll /api/status like the page does, until the job is no longer running."""
    end = time.monotonic() + budget_s
    while time.monotonic() < end:
        _, status = call(f"{base}/api/status")
        if status["job"]["state"] != "running":
            return status
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")
