"""The only retry loop in hebzip (the no-hand-made-retry rule: no hand-rolled retry loops).

``with_retry`` + ``locked_file``, trimmed to what
files need. ``locked_file`` is of Windows: ``os.replace`` fails with ``PermissionError``
for a moment while another process holds the file. Only that error is retried (40 x 50 ms).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

LOCKED_FILE_TRIES = 40
LOCKED_FILE_DELAY = 0.05


def with_retry[T](
    action: Callable[[], T],
    *,
    retry_on: tuple[type[Exception], ...],
    delays: Sequence[float],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``action``; on a retryable error wait ``delays[i]`` and try again; last error re-raised."""
    for delay in delays:
        try:
            return action()
        except retry_on:
            sleep(delay)
    return action()


def locked_file[T](action: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
    """Run a file operation, waiting out Windows' momentary "file in use"."""
    delays = [LOCKED_FILE_DELAY] * (LOCKED_FILE_TRIES - 1)
    return with_retry(action, retry_on=(PermissionError,), delays=delays, sleep=sleep)
