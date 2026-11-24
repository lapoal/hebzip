"""The well-known compressors from the Python standard library: gzip (ZIP), xz, bzip2.

One small class per family; each validates its own ``options`` row (unknown key = error at
startup), so a level is a config row and not a number in code (the change-without-code rule).
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import platform
import zlib
from pathlib import Path
from typing import Any

from atar.core.errors import ConfigError, ToolFailed


def check_options(options: dict[str, Any], allowed: dict[str, type], kind: str) -> dict[str, Any]:
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ConfigError(f"בהגדרות: אפשרות לא מוכרת ל-{kind}: {', '.join(unknown)}")
    for key, typ in allowed.items():
        if key not in options or not isinstance(options[key], typ):
            raise ConfigError(f"בהגדרות: חסרה או שגויה האפשרות '{key}' של {kind}")
    return options


class _Bytes:
    """Shared shape: read whole file, transform, write. Errors become ToolFailed."""

    def _pack(self, data: bytes) -> bytes:
        raise NotImplementedError

    def _unpack(self, data: bytes) -> bytes:
        raise NotImplementedError

    def prepare(self) -> None:
        """In-process: always ready."""

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        dst.write_bytes(self._pack(src.read_bytes()))

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        try:
            dst.write_bytes(self._unpack(src.read_bytes()))
        except (OSError, EOFError, ValueError, lzma.LZMAError, zlib.error) as error:
            raise ToolFailed(f"הפתיחה נכשלה ({type(error).__name__})") from error


class GzipCompressor(_Bytes):
    """ZIP's deflate in a gzip wrapper; mtime=0 so the same text gives the same bytes."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.level = int(check_options(options, {"level": int}, "gzip")["level"])

    def version(self) -> str:
        return f"zlib {zlib.ZLIB_RUNTIME_VERSION}, level {self.level}"

    def _pack(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level, mtime=0)

    def _unpack(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class XzCompressor(_Bytes):
    """xz (LZMA2); ``extreme`` adds the -e flag."""

    def __init__(self, options: dict[str, Any]) -> None:
        opts = check_options(options, {"preset": int, "extreme": bool}, "xz")
        self.preset = int(opts["preset"]) | (lzma.PRESET_EXTREME if opts["extreme"] else 0)
        self.label = f"{opts['preset']}{'e' if opts['extreme'] else ''}"

    def version(self) -> str:
        return f"Python {platform.python_version()} lzma, -{self.label}"

    def _pack(self, data: bytes) -> bytes:
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=self.preset)

    def _unpack(self, data: bytes) -> bytes:
        return lzma.decompress(data, format=lzma.FORMAT_XZ)


class Bz2Compressor(_Bytes):
    """bzip2 at the configured block level."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.level = int(check_options(options, {"level": int}, "bz2")["level"])

    def version(self) -> str:
        return f"Python {platform.python_version()} bz2, level {self.level}"

    def _pack(self, data: bytes) -> bytes:
        return bz2.compress(data, compresslevel=self.level)

    def _unpack(self, data: bytes) -> bytes:
        return bz2.decompress(data)
