"""Process helpers for the tests: is a pid alive, which processes a pid started, wait until gone.

Why ask Windows and not ``Popen``: the venv's ``python.exe`` is a launcher whose CHILD is the real
interpreter - "no process left" must look at the child too.
"""

from __future__ import annotations

import subprocess
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def alive(pid: int) -> bool:
    done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True,
                          text=True, timeout=60, check=False, creationflags=NO_WINDOW)
    return f'"{pid}"' in done.stdout


def children(pid: int) -> list[int]:
    query = f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"
    done = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True,
                          timeout=60, check=False, creationflags=NO_WINDOW)
    return [int(word) for word in done.stdout.split() if word.isdigit()]


def tree(pid: int, depth: int = 6) -> list[int]:
    """``pid`` and everything below it (window -> cmd -> launcher -> interpreter -> service -> ...)."""
    out = [pid]
    if depth > 0:
        for child in children(pid):
            out += tree(child, depth - 1)
    return out


CLOSE_WINDOW = """
import ctypes, sys
k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
k32.FreeConsole()
if not k32.AttachConsole(int(sys.argv[1])):
    raise SystemExit(2)
hwnd = k32.GetConsoleWindow()
k32.FreeConsole()
raise SystemExit(0 if hwnd and u32.PostMessageW(hwnd, 0x0010, 0, 0) else 3)
"""
"""WM_CLOSE to the console window of process ``argv[1]`` - what the window's X button sends. Run in its own
interpreter so the test process keeps its console."""


def close_console_window(pid_in_console: int, python: str) -> int:
    """Close the (classic ``conhost``) window that ``pid_in_console`` is attached to, like the user does."""
    done = subprocess.run([python, "-c", CLOSE_WINDOW, str(pid_in_console)], capture_output=True, timeout=60,
                          check=False, creationflags=NO_WINDOW)
    return done.returncode


def gone(pids: list[int], budget_s: float = 30) -> bool:
    end = time.monotonic() + budget_s
    while time.monotonic() < end:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.5)
    return False


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False,
                   creationflags=NO_WINDOW)
