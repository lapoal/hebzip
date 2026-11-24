"""Run one tool on the job's text: compress, decompress, compare bytes, time both directions.

"נפתח זהה למקור" is earned here only - by actually decompressing in a separate step and
comparing every byte with the original (the ready-only-after-proof rule: a mark only after proof).
Packages and decompressed copies stay inside the job's scratch folder.
"""

from __future__ import annotations

from pathlib import Path

from atar.compressors.contract import Compressor
from atar.config.load import ToolRow
from atar.core.clock import monotonic
from atar.core.errors import ToolFailed
from atar.core.logging import get_logger
from atar.domain.report import Outcome

_log = get_logger("runner")


def _failed(row: ToolRow, status: str, c_s: float | None = None) -> Outcome:
    return Outcome(row.id, row.label, row.ours, None, c_s, None, None, status)


def _timed_compress(tool: Compressor, text: Path, package: Path, folder: Path) -> float:
    tool.prepare()  # a loaded tool's one-time load is not the request's time (shown on its own line)
    start = monotonic()
    tool.compress(text, package, folder)
    return round(monotonic() - start, 2)


def run_tool(row: ToolRow, tool: Compressor, text: Path, work: Path) -> Outcome:
    """One row of the table. Never raises for a tool's failure - it becomes the row status."""
    folder = work / row.id
    folder.mkdir()
    package, restored = folder / "package.bin", folder / "restored.bin"
    try:
        c_s = _timed_compress(tool, text, package, folder)
    except ToolFailed as error:
        _log.warning("compress failed", tool=row.id, status=error.status)
        return _failed(row, error.status)
    size = package.stat().st_size
    try:
        tool.prepare()
        start = monotonic()
        tool.decompress(package, restored, folder)
    except ToolFailed as error:
        _log.warning("decompress failed", tool=row.id, status=error.status)
        return Outcome(row.id, row.label, row.ours, size, c_s, None, False, "mismatch")
    d_s = round(monotonic() - start, 2)
    identical = restored.is_file() and restored.read_bytes() == text.read_bytes()
    _log.info("tool done", tool=row.id, size=size, compress_s=c_s, decompress_s=d_s,
              identical=identical)
    return Outcome(row.id, row.label, row.ours, size, c_s, d_s, identical,
                   "ok" if identical else "mismatch")
