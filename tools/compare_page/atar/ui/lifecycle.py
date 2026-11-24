"""Why did the server stop? Start and stop lines with a reason, in the log.

Why: the page went down in the middle of the user's job and the log could not say when or why. Now:
* ``started`` writes ``logs/server.running`` (the pid only) and logs "server started";
* a normal end (Ctrl+C, the serve loop ending, an unexpected error) logs "server stopped" + reason and
  removes the marker;
* closing the window / logoff / shutdown: Windows calls a console handler first - it logs the reason
  ("window_closed" ...) and removes the marker before the process is ended;
* a forced kill (Task Manager, ``taskkill /F``) or a power cut cannot write anything - so the NEXT start
  finds the marker still there and logs "previous run ended without a stop line".
Numbers and fixed words only - never text, never file names.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

from atar.core.logging import get_logger

_log = get_logger("server")
_EVENTS = {0: "ctrl_c", 1: "ctrl_break", 2: "window_closed", 5: "logoff", 6: "system_shutdown"}
_keep: list[Any] = []  # the ctypes callback must stay alive as long as the process


def _marker(logs: Path) -> Path:
    return logs / "server.running"


def started(logs: Path, pid: int, port: int, services: int) -> None:
    """Log the start (and an unclean end of the previous run, if its marker is still there)."""
    marker = _marker(logs)
    if marker.exists():
        _log.warning("previous run ended without a stop line (killed, crashed or power cut)",
                     previous_pid=marker.read_text(encoding="ascii", errors="replace").strip())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(pid), encoding="ascii")
    _log.info("server started", port=port, pid=pid, services=services)


def stopped(logs: Path, reason: str) -> None:
    """Log the end with its reason and remove the marker."""
    _log.info("server stopped", reason=reason)
    _marker(logs).unlink(missing_ok=True)


def on_console_close(logs: Path) -> None:
    """Windows: log window close / logoff / shutdown before the process is ended (then let it end).
    The loaded services need nothing here: their stdin closes with this process and they end."""
    if sys.platform != "win32":
        return

    def handler(event: int) -> int:
        reason = _EVENTS.get(event, f"console_event_{event}")
        if event in (0, 1):
            return 0  # Ctrl+C / Ctrl+Break: Python's own handling (KeyboardInterrupt) logs it
        stopped(logs, reason)
        return 0

    callback = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(handler)
    _keep.append(callback)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, 1)
