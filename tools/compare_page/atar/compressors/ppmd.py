"""PPMd7 (variant H, the one 7-Zip uses) through pyppmd, a dependency of hebzip's generic fallback.

Why H and not "I8" (a known pyppmd decoding fault): pyppmd's PPMd8 at order 8 sometimes writes
a stream it then cannot read ("Corrupted input data"). PPMd7's decoder needs the original
length, so the package is ``4-byte little-endian length + stream`` - the 4 bytes are counted
in the size, as 7-Zip also stores the size in its header. The roundtrip is still verified
for every job by the runner; this file does not trust itself.
"""

from __future__ import annotations

import struct
from importlib import metadata
from pathlib import Path
from typing import Any

import pyppmd

from atar.compressors.stdlib import check_options
from atar.core.errors import ToolFailed

_HEADER = struct.Struct("<I")


class Ppmd7Compressor:
    """PPMd7 with the configured order and model memory."""

    def __init__(self, options: dict[str, Any]) -> None:
        opts = check_options(options, {"order": int, "memory_mb": int}, "ppmd7")
        self.order = int(opts["order"])
        self.memory = int(opts["memory_mb"]) << 20

    def version(self) -> str:
        return (f"pyppmd {metadata.version('pyppmd')} PPMd7, "
                f"order {self.order}, {self.memory >> 20} MB")

    def prepare(self) -> None:
        """In-process: always ready."""

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        data = src.read_bytes()
        encoder = pyppmd.Ppmd7Encoder(self.order, self.memory)
        stream = encoder.encode(data) + encoder.flush(endmark=False)
        dst.write_bytes(_HEADER.pack(len(data)) + stream)

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        package = src.read_bytes()
        if len(package) < _HEADER.size:
            raise ToolFailed("החבילה קצרה מדי")
        (length,) = _HEADER.unpack_from(package)
        decoder = pyppmd.Ppmd7Decoder(self.order, self.memory)
        try:
            dst.write_bytes(bytes(decoder.decode(package[_HEADER.size:], length)))
        except (ValueError, pyppmd.PpmdError) as error:
            raise ToolFailed(f"הפתיחה נכשלה ({type(error).__name__})") from error
