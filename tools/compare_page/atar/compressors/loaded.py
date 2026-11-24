"""A tool that stays loaded: one long-lived child process, JSON lines both ways.

Why: our CLI spends ~3 s per call before it codes a single word (interpreter, imports, building
the mode, numba's cache) - on a short text that is 95% of the row's time. Its
``python -m hebzip serve`` builds the modes once and then answers requests; this class owns that process.

The long-lived-child pattern: (1) every
request has a time budget and the answer is awaited on a QUEUE fed by a reader thread, never on the
pipe (a Windows pipe cannot be read with a timeout); (2) a child that died or overran is killed and
started again on the NEXT use - a request is never silently retried; (4) tested with a fake child in a
real process (``tests/fake_service.py``) and with the real hebzip service. The protocol is
JSON lines with file paths; see also the Windows venv trap below.

* Start in the background (never block a request): ``start()`` returns at once; the page shows "loading"
  and an upload that arrives before "ready" waits in ``ensure_ready`` instead of failing.
* Stop: closing stdin is the event - the service ends by itself (also when this process is killed:
  its end of the pipe closes). ``stop()`` closes stdin, waits ``STOP_GRACE_S``, then kills the TREE:
  the venv's ``python.exe`` is a launcher whose child is the real interpreter.
* Privacy (the machine-and-material rule): only file paths inside the job's scratch folder cross the pipe; the
  child's stderr goes to the null device, and the state shown on the page is fixed words and numbers.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import IO, Any

from atar.config.load import ServiceRow
from atar.core.clock import monotonic
from atar.core.errors import ConfigError, ToolFailed
from atar.core.logging import get_logger

_log = get_logger("loaded")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
STOP_GRACE_S = 10.0
NOT_LOADED = "נכשל (הכלי לא נטען)"
FELL = "נכשל (הכלי הטעון נפל)"
Launch = Callable[[ServiceRow], "subprocess.Popen[bytes]"]


def launch(row: ServiceRow) -> subprocess.Popen[bytes]:
    """The configured command, its own (hidden) console, pipes for stdin/stdout, stderr discarded."""
    return subprocess.Popen(list(row.command), cwd=row.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """End the process and everything under it (the launcher's real interpreter too)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False,
                       creationflags=_NO_WINDOW)
    else:
        proc.kill()
    try:
        proc.wait(STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        _log.error("service did not end after kill", pid=proc.pid)


def _pump(stream: IO[bytes], into: queue.Queue[bytes | None]) -> None:
    for line in iter(stream.readline, b""):
        into.put(line)
    into.put(None)


def _decode(line: bytes | None) -> dict[str, Any]:
    """A reply line as a dict; anything else (closed, not JSON) -> ``{}``."""
    if line is None:
        return {}
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class LoadedService:
    """One service process: loading -> ready -> (stopped | failed), restarted on the next use."""

    def __init__(self, row: ServiceRow, launcher: Launch = launch) -> None:
        self.row = row
        self._launch = launcher
        self._state = threading.Lock()
        self._asking = threading.Lock()
        self._ready = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._id = 0
        self.state = "stopped"  # stopped | loading | ready | failed
        self.error = ""
        self.load_s: float | None = None
        self.starts = 0
        self.modes: list[str] = []

    def start(self) -> None:
        """Begin loading in the background (returns at once); nothing to do if loading or ready."""
        with self._state:
            if not self._usable():
                self._spawn()

    def _usable(self) -> bool:
        alive = self._proc is not None and self._proc.poll() is None
        return alive and self.state in ("loading", "ready")

    def _spawn(self) -> None:
        """(state lock held) A fresh process + a thread that waits for its ready line."""
        if self._proc is not None:
            kill_tree(self._proc)
        self._ready.clear()
        self.state, self.error, self.load_s, self.modes = "loading", "", None, []
        self._lines = queue.Queue()
        try:
            self._proc = self._launch(self.row)
        except OSError:
            self._proc = None
            self._settle(None, "failed", "התוכנה של הכלי לא נמצאה או לא נפתחה")
            return
        self.starts += 1
        assert self._proc.stdout is not None
        threading.Thread(target=_pump, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=self._await_ready, args=(self._proc, self._lines, monotonic()),
                         daemon=True).start()

    def _await_ready(self, proc: subprocess.Popen[bytes], lines: queue.Queue[bytes | None],
                     started: float) -> None:
        try:
            first = _decode(lines.get(timeout=self.row.ready_timeout_s))
        except queue.Empty:
            first = {"error": "הכלי לא סיים להיטען בזמן שהוקצב"}
        with self._state:
            if proc is not self._proc:
                return  # an older process; a newer one owns the state now
            if first.get("ready") is True:
                self.modes = [str(m) for m in first.get("modes", [])]
                self._settle(round(monotonic() - started, 2), "ready", "")
                _log.info("service ready", service=self.row.name, load_s=self.load_s, starts=self.starts)
                return
            kill_tree(proc)
            self._settle(None, "failed", str(first.get("error") or "הכלי נסגר בזמן הטעינה"))
            _log.error("service did not load", service=self.row.name, exit=proc.returncode)

    def _settle(self, load_s: float | None, state: str, error: str) -> None:
        self.load_s, self.state, self.error = load_s, state, error
        self._ready.set()

    def ensure_ready(self) -> None:
        """Wait until loaded - starting it again if it is not running; ``ToolFailed`` if it cannot load."""
        self.start()
        if not self._ready.wait(self.row.ready_timeout_s + STOP_GRACE_S):
            raise ToolFailed("הכלי לא סיים להיטען בזמן", NOT_LOADED)
        with self._state:
            if self.state != "ready":
                raise ToolFailed(self.error or "הכלי לא נטען", NOT_LOADED)

    def ask(self, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """One request -> its reply (``ok`` may be false: that is the tool's answer, not a crash)."""
        with self._asking:
            self.ensure_ready()
            with self._state:
                proc, lines = self._proc, self._lines
                self._id += 1
                rid = self._id
            assert proc is not None and proc.stdin is not None
            try:
                proc.stdin.write(json.dumps({**request, "id": rid}).encode("ascii") + b"\n")
                proc.stdin.flush()
                reply = _decode(lines.get(timeout=timeout_s))
            except queue.Empty:
                self._lost(proc, "stopped", "")
                raise ToolFailed("הכלי חרג מהזמן שהוקצב לו", "נעצר (חרג מהזמן)") from None
            except OSError:
                reply = {}
            if reply.get("id") != rid:
                self._lost(proc, "stopped", "הכלי הטעון נפל; ייטען מחדש בשימוש הבא")
                raise ToolFailed("הכלי הטעון נפל באמצע הבקשה", FELL)
            return reply

    def _lost(self, proc: subprocess.Popen[bytes], state: str, error: str) -> None:
        """The process overran or died: kill what is left; the next use starts a new one."""
        kill_tree(proc)
        with self._state:
            if proc is self._proc:
                self.state, self.error, self.load_s = state, error, None
                _log.error("service lost", service=self.row.name, state=state)

    def stop(self) -> None:
        """Close stdin (the service ends by itself), wait a little, then kill the tree if needed."""
        with self._state:
            proc, self._proc = self._proc, None
            self.state = "stopped"
            self._ready.set()
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(STOP_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            _log.warning("service did not stop by itself", service=self.row.name)
        kill_tree(proc)

    def pid(self) -> int | None:
        """The (launcher) process id while running - for tests and measurements."""
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def snapshot(self) -> dict[str, Any]:
        """What the page shows: fixed words and numbers only."""
        with self._state:
            return {"name": self.row.name, "version": self.row.version, "state": self.state,
                    "load_s": self.load_s, "modes": list(self.modes), "error": self.error,
                    "starts": self.starts}


class ServicePool:
    """The configured services; one is created only when a row takes it (so a row switched back to
    ``command`` loads nothing), and the pool starts / stops exactly those."""

    def __init__(self, rows: tuple[ServiceRow, ...], launcher: Launch = launch) -> None:
        self._rows = {row.name: row for row in rows}
        self._launch = launcher
        self.used: dict[str, LoadedService] = {}

    def take(self, name: str) -> LoadedService:
        """The service ``name`` (one object for every row that uses it)."""
        if name not in self._rows:
            raise ConfigError(f"בהגדרות: אין שירות בשם '{name}' בפרק [services]")
        if name not in self.used:
            self.used[name] = LoadedService(self._rows[name], self._launch)
        return self.used[name]

    def start_all(self) -> None:
        for service in self.used.values():
            service.start()

    def stop_all(self) -> None:
        for service in self.used.values():
            service.stop()

    def snapshot(self) -> list[dict[str, Any]]:
        return [service.snapshot() for service in self.used.values()]
