"""The one place that reads time - a stopwatch for timings, and the wall clock for log lines."""

from __future__ import annotations

import time
from datetime import datetime


def monotonic() -> float:
    """Seconds from an arbitrary start; only differences are meaningful."""
    return time.perf_counter()


def stamp() -> str:
    """Local wall-clock time for a log line, to the second (without it nobody
    could tell when or why the server closed)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
