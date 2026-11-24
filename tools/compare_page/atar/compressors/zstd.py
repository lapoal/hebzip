"""Zstandard through the ``zstandard`` module (a dependency of hebzip's generic fallback).

Why a drawer of its own (the public measurement): zstd is the modern general-purpose
compressor people compare with; the module is already installed with hebzip, so no new dependency.
The frame keeps its content size (zstd's default), so the package decodes without being told the
length; no checksum (``write_checksum`` off, like ``zstd`` without ``--check``).

Why a second drawer with a trained dictionary: hebzip ships prior knowledge built from a Wikipedia
sample, while plain zstd starts empty - so the fair question is what zstd does when it, too, is handed
knowledge from the same sample. ``ZstdDictCompressor`` compresses every text alone with one dictionary
file (trained by ``atar.bench.zdict``), like ``zstd -19 -D dict``: the frame carries the dictionary id
(zstd's default), and the dictionary itself is the row's prior knowledge. The file is read on first
use; a missing or unreadable one is a failed row (``ToolFailed``), never a size it did not earn.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

import zstandard

from atar.compressors.stdlib import check_options
from atar.core.errors import ToolFailed


def library() -> str:
    """The module and the C library it runs - one string for every zstd row and the dictionary record."""
    libzstd = ".".join(map(str, zstandard.ZSTD_VERSION))
    return f"zstandard {metadata.version('zstandard')} (libzstd {libzstd})"


class ZstdCompressor:
    """zstd at the configured level (1..22)."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.level = int(check_options(options, {"level": int}, "zstd")["level"])

    def version(self) -> str:
        return f"{library()}, level {self.level}"

    def prepare(self) -> None:
        """In-process: always ready."""

    def _packer(self) -> zstandard.ZstdCompressor:
        return zstandard.ZstdCompressor(level=self.level)

    def _unpacker(self) -> zstandard.ZstdDecompressor:
        return zstandard.ZstdDecompressor()

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        dst.write_bytes(self._packer().compress(src.read_bytes()))

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        try:
            dst.write_bytes(self._unpacker().decompress(src.read_bytes()))
        except zstandard.ZstdError as error:
            raise ToolFailed(f"הפתיחה נכשלה ({type(error).__name__})") from error


class ZstdDictCompressor(ZstdCompressor):
    """zstd at the configured level with one trained dictionary (``dictionary`` = its file)."""

    def __init__(self, options: dict[str, Any]) -> None:
        opts = check_options(options, {"level": int, "dictionary": str}, "zstd_dict")
        self.level = int(opts["level"])
        self.path = Path(opts["dictionary"])
        self._raw: bytes | None = None

    def _dictionary(self) -> zstandard.ZstdCompressionDict:
        if self._raw is None:
            try:
                self._raw = self.path.read_bytes()
            except OSError as error:
                raise ToolFailed(f"קובץ המילון חסר ({type(error).__name__})", "אין מילון") from error
        return zstandard.ZstdCompressionDict(self._raw)

    def version(self) -> str:
        """Names the dictionary by size and sha256 - never by its path (the results are public)."""
        if not self.path.is_file():
            return f"{library()}, level {self.level}, dictionary missing"
        raw = self.path.read_bytes()
        return (f"{library()}, level {self.level}, trained dictionary {len(raw):,} B "
                f"(sha256 {hashlib.sha256(raw).hexdigest()[:12]})")

    def prepare(self) -> None:
        """Ready once the dictionary file can be read."""
        self._dictionary()

    def _packer(self) -> zstandard.ZstdCompressor:
        return zstandard.ZstdCompressor(level=self.level, dict_data=self._dictionary())

    def _unpacker(self) -> zstandard.ZstdDecompressor:
        return zstandard.ZstdDecompressor(dict_data=self._dictionary())
