"""The fallback drawer ``best_of``: try every configured general-purpose codec, keep the smallest.

Frame = ``bytes([codec index]) + varint(len(raw)) + packed``; the codec list is a config row
(``[modes.generic].codecs``, its order = the index), each word ``family[:level][:option]``:
raw, deflate:L, bzip2:L, xz:L[:extreme], zstd:L, ppmd:V:ORDER (memory = ``ppmd_memory_mb``).
With the research prototype's 15 codecs in its order the frames are byte-identical to ``golden/generic``.
Why here (``integrations``): third-party compressors (zstandard, pyppmd) - the domain stays pure.
Ties keep the FIRST smallest codec, like the research prototype's ``min(range(...), key=len)``.

Optional libraries (the browser build): zstandard and pyppmd are imported the first time a codec
of theirs PACKS or UNPACKS (``_zstandard`` / ``_pyppmd``), never when this module is imported or the
codec list is built.
Why: the browser build (Pyodide) runs mode ``fast`` only and has neither wheel; importing them at module
load made the whole package unusable there. A missing library is a typed refusal naming the package
(``DrawerUnavailable``, exit 6), never a guess or a silently shorter codec list (the codec byte is the
index in the configured list, so dropping an entry would change packages). Same bytes when installed.
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from collections.abc import Callable
from typing import Any

from hebzip.core.errors import CorruptPackage, DrawerUnavailable
from hebzip.domain.varint import decode_varint, encode_varint

Pack = Callable[[bytes], bytes]
Unpack = Callable[[bytes, int], bytes]
DAMAGE: tuple[type[BaseException], ...] = (OSError, EOFError, RuntimeError, ValueError, SystemError, zlib.error,
                                           lzma.LZMAError)
"""Everything the codecs raise on damaged data - turned into ``CorruptPackage``. ``SystemError``: the
ppmd C decoder returns NULL without an exception on some damaged frames. zstandard's own ``ZstdError`` is
caught inside its unpacker (``_zstd``), which is the only place that has the module."""


def _missing(family: str, package: str) -> DrawerUnavailable:
    return DrawerUnavailable(f"הקודק {family} צריך את חבילת הפייתון {package}, שאינה מותקנת "
                             f"(the {family} codec needs the Python package {package}: pip install {package})")


def _zstandard() -> Any:
    """zstandard, imported now (optional, see the module text); missing -> ``DrawerUnavailable`` naming it."""
    try:
        import zstandard
    except ImportError as error:
        raise _missing("zstd", "zstandard") from error
    return zstandard


def _pyppmd() -> Any:
    """pyppmd, imported now (optional, see the module text); missing -> ``DrawerUnavailable`` naming it."""
    try:
        import pyppmd
    except ImportError as error:
        raise _missing("ppmd", "pyppmd") from error
    return pyppmd


Codec = tuple[Pack, Unpack]
XZ_FLAGS: dict[str, int] = {"extreme": lzma.PRESET_EXTREME}
"""xz option word -> preset bit."""


def _raw(options: list[str], memory: int) -> Codec:
    return (lambda raw: raw), (lambda data, n: data)


def _deflate(options: list[str], memory: int) -> Codec:
    level = int(options[0])
    return (lambda raw: zlib.compress(raw, level)), (lambda data, n: zlib.decompress(data))


def _bzip2(options: list[str], memory: int) -> Codec:
    level = int(options[0])
    return (lambda raw: bz2.compress(raw, level)), (lambda data, n: bz2.decompress(data))


def _xz(options: list[str], memory: int) -> Codec:
    preset = int(options[0])
    for flag in options[1:]:
        preset |= XZ_FLAGS[flag]
    return (lambda raw: lzma.compress(raw, preset=preset)), (lambda data, n: lzma.decompress(data))


def _zstd(options: list[str], memory: int) -> Codec:
    level = int(options[0])

    def pack(raw: bytes) -> bytes:
        return bytes(_zstandard().ZstdCompressor(level=level).compress(raw))

    def unpack(data: bytes, n: int) -> bytes:
        zstandard = _zstandard()
        try:
            return bytes(zstandard.ZstdDecompressor().decompress(data))
        except zstandard.ZstdError as error:
            raise CorruptPackage(f"פענוח כללי נכשל ({type(error).__name__})") from error

    return pack, unpack


def _ppmd7_unpack(order: int, memory: int) -> Unpack:
    return lambda data, n: bytes(_pyppmd().Ppmd7Decoder(order, memory).decode(data, n))


def _ppmd8_unpack(order: int, memory: int) -> Unpack:
    def unpack(data: bytes, n: int) -> bytes:
        out = _pyppmd().decompress(data, max_order=order, mem_size=memory, variant="I")
        if not isinstance(out, bytes):
            raise CorruptPackage("פענוח כללי נכשל (ppmd)")
        return out
    return unpack


PPMD_UNPACK: dict[str, Callable[[int, int], Unpack]] = {"H": _ppmd7_unpack, "I": _ppmd8_unpack}
"""ppmd variant word -> its decoder (H = PPMd7 with the size-bounded decoder, I = PPMd8)."""


def _ppmd(options: list[str], memory_mb: int) -> Codec:
    variant, order, memory = options[0], int(options[1]), memory_mb << 20

    def pack(raw: bytes) -> bytes:
        return bytes(_pyppmd().compress(raw, max_order=order, mem_size=memory, variant=variant))

    return pack, PPMD_UNPACK[variant](order, memory)


CODECS: dict[str, Callable[[list[str], int], Codec]] = {
    "raw": _raw, "deflate": _deflate, "bzip2": _bzip2, "xz": _xz, "zstd": _zstd, "ppmd": _ppmd}
"""Codec family -> (options after the family, ``ppmd_memory_mb``) -> (pack, unpack). The selector resolves
each word of ``[modes.<g>].codecs`` here (no if-chain on the family); the config accepts exactly
these families (``schema.CODEC_FORMS`` - tested to be the same keys)."""


class BestOfCodec:
    """``best_of``: every configured codec, smallest frame wins."""

    name = "best_of"

    def __init__(self, names: list[str], codecs: list[Codec]) -> None:
        self.names = list(names)
        self.codecs = list(codecs)

    def encode(self, raw: bytes) -> bytes:
        """Raw -> codec byte + varint length + packed."""
        packed = [pack(raw) for pack, _ in self.codecs]
        best = min(range(len(packed)), key=lambda i: len(packed[i]))
        return bytes([best]) + encode_varint(len(raw)) + packed[best]

    def decode(self, payload: bytes) -> bytes:
        """Frame -> raw; a cut frame or unknown codec byte -> ``CorruptPackage``."""
        if not payload or payload[0] >= len(self.codecs):
            raise CorruptPackage("מסגרת כללית פגומה (קודק לא מוכר)")
        n, pos = decode_varint(payload, 1)
        try:
            raw = self.codecs[payload[0]][1](payload[pos:], n)
        except DAMAGE as error:
            raise CorruptPackage(f"פענוח כללי נכשל ({type(error).__name__})") from error
        if len(raw) != n:
            raise CorruptPackage("פענוח כללי נכשל (אורך שונה מהמוצהר)")
        return raw

    def codec_names(self) -> list[str]:
        """The configured codec words."""
        return list(self.names)
