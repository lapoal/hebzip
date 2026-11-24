"""How much memory Windows can still promise a new process (commit headroom), in MB.

Why commit and not "free RAM": paq8px -8L reserves ~2.2 GB up front; on this
laptop free RAM is often 1-4 GB while the page file still covers the reservation, and
paq8px ran fine. What actually fails is the commit (RAM + page file) running out, so that
is what the guard measures. Unknown (not Windows) -> a huge number: never block a run
on a measurement we could not make (the cannot-measure-is-not-invalid rule).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _status() -> _MemoryStatus | None:
    if sys.platform != "win32":
        return None
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status


def available_commit_mb() -> int:
    """Commit headroom in MB (a very large number when it cannot be measured)."""
    status = _status()
    return 1 << 30 if status is None else int(status.ullAvailPageFile >> 20)


def total_ram_mb() -> int:
    """Installed physical memory in MB (0 when it cannot be measured) - for the bench's machine line."""
    status = _status()
    return 0 if status is None else int(status.ullTotalPhys >> 20)
