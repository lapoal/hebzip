"""One tool on one text: compress, decompress in a separate call, compare every byte, time both.

Why its own loop and not the page's ``services.runner.run_tool``: the page rounds times to 0.01 s for
display, while a published median of short texts needs milliseconds; and the bench keeps the sha256 of
every package (so the loaded service and the cold CLI can be proven to write the same bytes). The
drawers are the page's (``atar.compressors``) - one implementation of every tool.
"Exact" is earned only by decompressing and comparing bytes (the ready-only-after-proof rule).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atar.bench.config import BenchRow
from atar.compressors.contract import Compressor
from atar.core.clock import monotonic
from atar.core.errors import ToolFailed

HEBZIP_MODE_BITS = 0x3F
"""A compact hebzip header's first byte = 0x80 | mode id (``hebzip/domain/header.py``)."""


@dataclass(frozen=True, slots=True)
class Measured:
    """The numbers of one (tool, text) pair - never the text."""

    tool: str
    set_id: str
    text: str
    raw: int
    bytes: int
    exact: bool
    package_sha256: str
    compress_s: float
    decompress_s: float
    mode_id: int
    status: str

    def as_json(self) -> dict[str, Any]:
        return {"tool": self.tool, "set": self.set_id, "text": self.text, "raw": self.raw,
                "bytes": self.bytes, "exact": self.exact, "package_sha256": self.package_sha256,
                "compress_s": self.compress_s, "decompress_s": self.decompress_s, "mode_id": self.mode_id,
                "status": self.status}


def _timed(call: Any, *args: Path) -> float:
    start = monotonic()
    call(*args)
    return round(monotonic() - start, 4)


def measure(row: BenchRow, tool: Compressor, set_id: str, name: str, text: Path, folder: Path) -> Measured:
    """One pair; a tool's failure becomes the status (``failed: ...``), never an exception."""
    package, restored = folder / "package.bin", folder / "restored.bin"
    raw = text.stat().st_size
    tool.prepare()  # a loaded tool's one-time load is not part of any text's time
    try:
        c_s = _timed(tool.compress, text, package, folder)
        tool.prepare()
        d_s = _timed(tool.decompress, package, restored, folder)
    except ToolFailed as error:
        return Measured(row.tool.id, set_id, name, raw, -1, False, "", -1.0, -1.0, 0,
                        f"failed: {error.status}")
    data = package.read_bytes()
    exact = restored.is_file() and restored.read_bytes() == text.read_bytes()
    mode_id = data[0] & HEBZIP_MODE_BITS if row.tool.ours and data else 0
    status = "ok" if exact else "mismatch"
    if exact and row.mode_id and mode_id != row.mode_id:
        status = "ok (fallback)"
    return Measured(row.tool.id, set_id, name, raw, len(data), exact, hashlib.sha256(data).hexdigest(),
                    c_s, d_s, mode_id, status)
