"""A test-side client for ``python -m hebzip serve`` (the real entry point in a child process).

Why a real child (the real-path rule; child processes are tested as real processes): the page
talks to the service through pipes, so the tests do too. Replies are awaited on a QUEUE fed by a reader
thread, never on the pipe itself (a Windows pipe cannot be read with a timeout). stderr is drained by a
second thread into ``err`` - a full stderr pipe would block the service, and the canary scans it.
Windows trap: the venv's ``python.exe`` is a launcher whose CHILD is the
real interpreter; ``Popen.kill`` ends only the launcher, so ``kill_tree`` uses ``taskkill /T /F``.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any

from tests.support import CONFIG, ROOT

READY_TIMEOUT_S = 600
ASK_TIMEOUT_S = 900
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def serve_command(*modes: str, config: Path = CONFIG) -> list[str]:
    """The command line the comparison page uses."""
    cmd = [sys.executable, "-m", "hebzip", "serve", "--config", str(config)]
    return cmd + (["--modes", ",".join(modes)] if modes else [])


def _pump(stream: IO[bytes], into: queue.Queue[bytes | None]) -> None:
    for line in iter(stream.readline, b""):
        into.put(line)
    into.put(None)


def _drain(stream: IO[bytes], into: bytearray) -> None:
    for chunk in iter(lambda: stream.read(4096), b""):
        into.extend(chunk)


class Service:
    """One running service; ``ready`` is its first line (``{"ready": ...}``)."""

    def __init__(self, *modes: str, config: Path = CONFIG) -> None:
        self.proc = subprocess.Popen(serve_command(*modes, config=config), cwd=ROOT, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        assert self.proc.stdout is not None and self.proc.stderr is not None
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self.err = bytearray()
        self.out = bytearray()
        threading.Thread(target=_pump, args=(self.proc.stdout, self.lines), daemon=True).start()
        threading.Thread(target=_drain, args=(self.proc.stderr, self.err), daemon=True).start()
        self._id = 0
        self.ready = self.next(READY_TIMEOUT_S)

    def next(self, timeout: float) -> dict[str, Any]:
        """The next line as JSON; ``{}`` when the service closed its stdout."""
        line = self.lines.get(timeout=timeout)
        if line is None:
            return {}
        self.out.extend(line)
        answer: dict[str, Any] = json.loads(line)
        return answer

    def send_line(self, line: bytes) -> None:
        """A raw line (also a broken one) to the service."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + b"\n")
        self.proc.stdin.flush()

    def ask(self, op: str, timeout: float = ASK_TIMEOUT_S, **fields: object) -> dict[str, Any]:
        """One request -> its reply (paths are sent as strings)."""
        self._id += 1
        body = {"id": self._id, "op": op, **{k: str(v) if isinstance(v, Path) else v for k, v in fields.items()}}
        self.send_line(json.dumps(body).encode("ascii"))
        return self.next(timeout)

    def compress(self, src: Path, dst: Path, mode: str) -> dict[str, Any]:
        """Compress like the CLI's ``compress --mode M --force``."""
        return self.ask("compress", input=src, output=dst, mode=mode, force=True)

    def decompress(self, src: Path, dst: Path) -> dict[str, Any]:
        """Decompress like the CLI's ``decompress --force``."""
        return self.ask("decompress", input=src, output=dst, force=True)

    def close(self, timeout: float = 60) -> int:
        """Close stdin (what the page does at its end) and wait for the exit code."""
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout)

    def kill_tree(self) -> None:
        """End the launcher AND the real interpreter under it."""
        kill_tree(self.proc.pid)
        self.proc.wait(30)

    def __enter__(self) -> Service:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc.poll() is None:
            try:
                self.close(60)
            except subprocess.TimeoutExpired:
                self.kill_tree()


def kill_tree(pid: int) -> None:
    """``taskkill /T /F`` on ``pid`` and everything under it."""
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False,
                   creationflags=NO_WINDOW)


def children(pid: int) -> list[int]:
    """The ids of the processes whose parent is ``pid`` (the real interpreter under a venv launcher)."""
    query = f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"
    done = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True,
                          timeout=60, check=False, creationflags=NO_WINDOW)
    return [int(word) for word in done.stdout.split() if word.isdigit()]


def alive(pid: int) -> bool:
    """Is a process with this id running right now?"""
    done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True,
                          text=True, timeout=60, check=False, creationflags=NO_WINDOW)
    return f'"{pid}"' in done.stdout
