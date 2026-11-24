"""Typed errors - the only exceptions atar code raises on purpose.

No bare
``except Exception`` except at the root safety net. Every message is Hebrew, because
it may reach the page, and never contains the uploaded text or the file name.
"""

from __future__ import annotations


class AtarError(Exception):
    """Base class for every deliberate atar failure (message = Hebrew, user-safe)."""


class ConfigError(AtarError):
    """The config file is missing, malformed, or has an unknown field - fail at startup."""


class InputRejected(AtarError):
    """The uploaded file cannot be compared (wrong type, not UTF-8, too large, empty)."""


class ToolFailed(AtarError):
    """One compressor did not produce a package or a decompressed file.

    ``status`` is the short Hebrew word the table shows for that row.
    """

    def __init__(self, message: str, status: str = "נכשל") -> None:
        super().__init__(message)
        self.status = status


class NotEnoughMemory(ToolFailed):
    """The tool needs more reserved memory than the machine can give right now."""

    def __init__(self) -> None:
        super().__init__("אין מספיק זיכרון פנוי לכלי הזה", "לא רץ (אין מספיק זיכרון)")


class Busy(AtarError):
    """A job is already running (one job at a time on a weak machine)."""
