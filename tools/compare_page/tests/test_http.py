"""Through the real HTTP path with fast tools: upload, poll, rows, busy, refusals, cleanup."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from atar.core.logging import configure_file_logging, stop_file_logging
from atar.services import material
from atar.services.jobs import JobBoard
from tests.support import (
    FAST,
    HEBREW,
    WORK,
    call,
    docx_bytes,
    live_server,
    paragraph,
    row,
    settings,
    upload,
    wait_done,
)

OURS = row("ours", "ppmd7", {"order": 4, "memory_mb": 16}, ours=True)
COPY = "import sys,shutil,time; time.sleep({s}); shutil.copy(sys.argv[1], sys.argv[2])"
LIAR = "import sys; open(sys.argv[2],'wb').write(b'not the same')"


def _cmd(tool_id: str, compress: str, decompress: str) -> Any:
    opts = {"version": "t", "cwd": "{work}", "memory_mb": 0, "memory_words": ["memory"],
            "compress": [sys.executable, "-c", compress, "{in}", "{out}"],
            "decompress": [sys.executable, "-c", decompress, "{in}", "{out}"]}
    return row(tool_id, "command", opts)


def test_upload_txt_fills_the_table() -> None:
    s = settings((*FAST, OURS))
    with live_server(s) as (base, _):
        code, _reply = upload(base, "מסמך.txt", HEBREW.encode("utf-8"))
        assert code == 202
        status = wait_done(base)
    job = status["job"]
    assert job["state"] == "done" and job["file_name"] == "מסמך.txt"
    assert job["original"] == len(HEBREW.encode("utf-8")) and job["words"] == len(HEBREW.split())
    sizes = [r["size"] for r in job["rows"]]
    assert sizes == sorted(sizes) and len(sizes) == 5
    assert all(r["status"] == "ok" and r["identical"] is True for r in job["rows"])
    assert job["summary"].startswith("הכלי שלנו")
    assert [v["label"] for v in status["versions"]] == ["zip", "bz2", "xz", "ppmd", "ours"]
    assert material.is_empty(s.material)


def test_docx_upload_counts_extracted_text_only() -> None:
    s = settings(FAST[:1])
    data = docx_bytes(f"<w:p>{paragraph('שלום עולם')}</w:p><w:p>{paragraph('שורה שנייה')}</w:p>")
    with live_server(s) as (base, _):
        assert upload(base, "a.docx", data)[0] == 202
        job = wait_done(base)["job"]
    assert job["original"] == len("שלום עולם\nשורה שנייה".encode()) and job["words"] == 4


def test_second_upload_while_busy_is_polite() -> None:
    s = settings((_cmd("slow", COPY.format(s=2), COPY.format(s=0)),))
    with live_server(s) as (base, _):
        assert upload(base, "a.txt", HEBREW.encode())[0] == 202
        code, reply = upload(base, "b.txt", HEBREW.encode())
        assert code == 409 and reply["error"].startswith("עסוק")
        _, mid = call(f"{base}/api/status")
        assert mid["job"]["running"] == "slow" and mid["job"]["pending"] == ["slow"]
        wait_done(base)
    assert material.is_empty(s.material)


@pytest.mark.parametrize(("name", "data", "code"), [
    ("a.pdf", b"x", 422), ("a.txt", b"\xff\xfe\x00", 422), ("a.txt", b"", 400),
    ("a.txt", "מילה ".encode() * 100, 422),
])
def test_refusals(name: str, data: bytes, code: int) -> None:
    s = settings(FAST[:1], max_text_bytes=100)
    with live_server(s) as (base, _):
        got, reply = upload(base, name, data)
        assert got == code and reply["error"]
        assert call(f"{base}/api/status")[1]["job"]["state"] == "idle"
    assert material.is_empty(s.material)


def test_mismatch_is_marked_and_scratch_is_wiped() -> None:
    s = settings((FAST[0], _cmd("liar", COPY.format(s=0), LIAR)))
    with live_server(s) as (base, _):
        upload(base, "a.txt", HEBREW.encode())
        job = wait_done(base)["job"]
    liar = next(r for r in job["rows"] if r["id"] == "liar")
    assert liar["status"] == "mismatch" and liar["identical"] is False
    assert material.is_empty(s.material)


def test_forced_job_crash_still_wipes_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: JobBoard, *args: object) -> None:
        raise RuntimeError("forced")

    monkeypatch.setattr(JobBoard, "_one", boom)
    s = settings(FAST[:2])
    with live_server(s) as (base, _):
        upload(base, "a.txt", HEBREW.encode())
        job = wait_done(base)["job"]
    assert job["state"] == "error" and "תקלה" in job["error"]
    assert material.is_empty(s.material)


def test_log_holds_no_text_and_no_file_name() -> None:
    canary = "קנרית" + "zq7x"
    s = settings(FAST[:2])
    configure_file_logging(WORK / "logs" / "atar.log")
    try:
        with live_server(s) as (base, _):
            upload(base, f"{canary}.txt", (HEBREW + canary).encode())
            wait_done(base)
    finally:
        stop_file_logging()
    log = (WORK / "logs" / "atar.log").read_text(encoding="utf-8")
    assert "tool done" in log and canary not in log and "zq7x" not in log


def test_static_page_has_csp_and_no_external_resources() -> None:
    import urllib.request
    s = settings(FAST[:1])
    with live_server(s) as (base, _), urllib.request.urlopen(base + "/", timeout=10) as reply:
        html = reply.read().decode("utf-8")
        assert "default-src 'self'" in reply.headers["Content-Security-Policy"]
    assert 'dir="rtl"' in html
    from atar.ui.app import STATIC
    for f in STATIC.iterdir():
        body = f.read_text(encoding="utf-8")
        assert "http://" not in body and "https://" not in body and "@import" not in body, f.name
