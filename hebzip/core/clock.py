"""The clock - the only place in hebzip that reads the time (the one-clock rule).

Only ``monotonic`` is needed here:
it measures durations for the log. Nothing in a package depends on the time - a package must be
byte-identical whenever it is made.
"""

from __future__ import annotations

import time


def monotonic() -> float:
    """A clock that never jumps back - for measuring how long something took."""
    return time.monotonic()
