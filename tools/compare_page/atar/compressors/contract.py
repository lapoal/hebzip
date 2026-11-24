"""The compressor socket: file -> package, package -> file, a version string.

Socket = contract + real drawers + a null plate (the socket rule (contract + real drawer + plate)). The null
plate refuses
every call with :class:`ToolFailed` - the safe direction: a row that is not wired can never
show a size or an "identical" mark it did not earn.

``work`` is the job's private scratch folder: a drawer may write only there (the machine-and-material rule);
it is deleted with everything in it when the job ends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atar.core.errors import ConfigError, ToolFailed


@runtime_checkable
class Compressor(Protocol):
    """What every row of the table needs from its tool."""

    def version(self) -> str:
        """A short version string for the note under the table."""
        ...

    def prepare(self) -> None:
        """Be ready to run (a tool that stays loaded waits for its load here); not timed.
        Raise ``ToolFailed`` if the tool cannot get ready. In-process tools: nothing to do."""
        ...

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        """Write the package of ``src`` to ``dst``; raise ``ToolFailed`` if it cannot."""
        ...

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        """Write the file restored from package ``src`` to ``dst``; raise ``ToolFailed``."""
        ...


class NullCompressor:
    """The plate: answers every call, does nothing, never claims a result."""

    def __init__(self, options: dict[str, object] | None = None) -> None:
        if options:
            raise ConfigError("בהגדרות: ללוחית (null) אין אפשרויות")

    def version(self) -> str:
        return "not wired"

    def prepare(self) -> None:
        raise ToolFailed("הכלי לא מחובר", "לא מחובר")

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        raise ToolFailed("הכלי לא מחובר", "לא מחובר")

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        raise ToolFailed("הכלי לא מחובר", "לא מחובר")
