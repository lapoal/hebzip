"""The real seven tools (real config rows) through HTTP, plus the canary gate (the machine-and-material rule).

A synthetic Hebrew text carrying a random marker goes through every real tool; afterwards
the marker must not exist anywhere outside the (now empty) material folder: not in this
tool's tree, not in files the repository's hebzip touched during the run, not in TEMP.
Slow on purpose (~25 s per upload): it is the path the user takes.
"""

from __future__ import annotations

import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from atar.config.load import REPO, ROOT
from atar.services import material
from tests.support import HEBREW, call, docx_bytes, live_server, paragraph, settings, upload, wait_done

SKIP = {".venv", "__pycache__", ".git"}


def _check_rows(job: dict[str, object]) -> list[dict[str, object]]:
    rows = job["rows"]
    assert isinstance(rows, list) and len(rows) == 8
    ids = {r["id"] for r in rows}
    assert ids == {"zip", "bz2", "xz", "ppmd", "paq8px", "fast", "max", "max_plus"}
    for r in rows:
        if r["id"] == "paq8px":
            assert r["status"] == "לא מחובר" and r["identical"] is not True, r  # not shipped: a plate row
            continue
        assert r["status"] == "ok" and r["identical"] is True, r
    return rows


def _touched_files(root: Path, since: float) -> list[Path]:
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            p = Path(folder, name)
            try:
                if p.stat().st_mtime >= since and p.stat().st_size < 50_000_000:
                    found.append(p)
            except OSError:
                continue
    return found


def _marker_anywhere(marker: bytes, since: float) -> list[str]:
    places = [ROOT, REPO, Path(tempfile.gettempdir())]
    hits = []
    for root in places:
        for p in _touched_files(root, since) if root != ROOT else _touched_files(root, 0):
            try:
                if marker in p.read_bytes():
                    hits.append(str(p))
            except OSError:
                continue
    return hits


def _check_service(status: dict[str, Any]) -> None:
    """Our rows really ran on the loaded hebzip service: loaded once, ready, all three modes."""
    (service,) = status["services"]
    assert service["state"] == "ready" and service["starts"] == 1, service
    assert service["modes"] == ["fast", "max", "max_plus"] and service["load_s"] > 0


def test_seven_real_tools_txt_and_docx_and_no_marker_left() -> None:
    """The first upload arrives while our tool is still loading: it waits, it does not fail. The marker
    scan covers the loaded service too: it runs in the repository tree (scanned), its stderr is discarded,
    and it keeps no file of its own."""
    marker = "סימן" + secrets.token_hex(6)
    s = settings()
    since = time.time() - 1
    with live_server(s) as (base, _):
        assert call(f"{base}/api/status")[1]["services"][0]["state"] == "loading"
        assert upload(base, "טקסט.txt", (HEBREW + marker + "\n").encode("utf-8"))[0] == 202
        done = wait_done(base)
        job = done["job"]
        _check_rows(job)
        _check_service(done)
        assert material.is_empty(s.material)
        body = f"<w:p>{paragraph(HEBREW.splitlines()[0])}</w:p><w:p>{paragraph(marker)}</w:p>"
        assert upload(base, "מסמך.docx", docx_bytes(body))[0] == 202
        done2 = wait_done(base)
        job2 = done2["job"]
        _check_rows(job2)
        _check_service(done2)
        assert job2["words"] == len(HEBREW.splitlines()[0].split()) + 1
    assert material.is_empty(s.material)
    assert _marker_anywhere(marker.encode("utf-8"), since) == []
