"""Zstandard through the ``zstandard`` module (a dependency of hebzip's generic fallback).

Why a drawer of its own (the public measurement): zstd is the modern general-purpose
compressor people compare with; the module is already installed with hebzip, so no new dependency.
The frame keeps its content size (zstd's default), so the package decodes without being told the
length; no checksum (``write_checksum`` off, like ``zstd`` without ``--check``).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import zstandard

from atar.compressors.stdlib import check_options
from atar.core.errors import ToolFailed


class ZstdCompressor:
    """zstd at the configured level (1..22)."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.level = int(check_options(options, {"level": int}, "zstd")["level"])

    def version(self) -> str:
        libzstd = ".".join(map(str, zstandard.ZSTD_VERSION))
        return f"zstandard {metadata.version('zstandard')} (libzstd {libzstd}), level {self.level}"

    def prepare(self) -> None:
        """In-process: always ready."""

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        dst.write_bytes(zstandard.ZstdCompressor(level=self.level).compress(src.read_bytes()))

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        try:
            dst.write_bytes(zstandard.ZstdDecompressor().decompress(src.read_bytes()))
        except zstandard.ZstdError as error:
            raise ToolFailed(f"הפתיחה נכשלה ({type(error).__name__})") from error
