"""Closing the start.bat window ends the page AND the loaded tool - no process left holding 1-2 GB.

The real ``start.bat`` runs in a classic console window (``conhost.exe``, so the test owns that one
window and never touches the user's Windows Terminal), with a test config (free port, material and
logs under ``tests/_work``). When the page says the tool is ready, the window gets WM_CLOSE - exactly
what its X button sends - and every process that was under the window must be gone, the service's
launcher and real interpreter included. Slow (~10-20 s: the real service loads).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Any

from atar.config.load import DEFAULT_CONFIG, ROOT
from tests.procs import alive, children, close_console_window, gone, kill_tree, tree
from tests.support import WORK, free_port


def _config(port: int) -> str:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    for old, new in (("port = 8837", f"port = {port}"), ("open_browser = true", "open_browser = false"),
                     ('material = "material"', f"material = '{WORK / 'material'}'"),
                     ('logs = "logs"', f"logs = '{WORK / 'logs'}'")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    cfg = WORK / "atar.toml"
    cfg.write_text(text, encoding="utf-8")
    return str(cfg)


def _status(port: int) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as reply:
            answer: dict[str, Any] = json.loads(reply.read())
            return answer
    except OSError:
        return None


def _wait_ready(port: int, budget_s: float = 600) -> dict[str, Any]:
    end = time.monotonic() + budget_s
    while time.monotonic() < end:
        status = _status(port)
        if status and status["services"] and status["services"][0]["state"] in ("ready", "failed"):
            return status
        time.sleep(0.5)
    raise AssertionError("the page never reported the tool ready")


def test_closing_the_window_ends_the_page_and_the_loaded_tool() -> None:
    port = free_port()
    window = subprocess.Popen(["conhost.exe", "cmd.exe", "/c", str(ROOT / "start.bat"), "--config",
                               _config(port), "--no-browser"], env={**os.environ, "HEBZIP_PYTHON":
                               sys.executable})
    procs: list[int] = []
    try:
        status = _wait_ready(port)
        assert status["services"][0]["state"] == "ready", status["services"]
        procs = tree(window.pid)
        assert len(procs) >= 6, procs  # conhost, cmd, launcher, page, service launcher, service
        shell = children(window.pid)[0]
        assert close_console_window(shell, sys.executable) == 0
        assert gone(procs), f"left running after the window closed: {[p for p in procs if alive(p)]}"
        assert _status(port) is None
    finally:
        for pid in procs or [window.pid]:
            kill_tree(pid)
    lines = _log_lines()
    assert all(line.get("ts") for line in lines), "every log line carries its time"
    assert any(line["msg"] == "server stopped" and line.get("reason") == "window_closed" for line in lines)
    assert not (WORK / "logs" / "server.running").exists()


def _log_lines() -> list[dict[str, Any]]:
    text = (WORK / "logs" / "atar.log").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_a_killed_server_is_reported_at_the_next_start() -> None:
    """A forced kill cannot write a stop line; the next start says the previous run ended uncleanly."""
    port = free_port()
    cfg = _config(port)
    for _ in range(2):
        proc = subprocess.Popen([sys.executable, "-m", "atar.ui.server", "--config", cfg, "--no-browser"],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            _wait_ready(port)
        finally:
            kill_tree(proc.pid)
            proc.wait(30)
        assert gone([proc.pid])
    lines = _log_lines()
    starts = [line for line in lines if line["msg"] == "server started"]
    unclean = [line for line in lines if line["msg"].startswith("previous run ended without a stop line")]
    assert len(starts) == 2 and len(unclean) == 1, lines
