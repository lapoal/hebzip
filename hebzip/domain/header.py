"""Package header, format 1 - COMPACT variant.

Why compact: the design chose +1 byte per package over the 4-byte header (a 174-byte text pays
0.6% instead of 2.3%). Layout, zero I/O:

  compact   [0x80 | mode_id]  [table_sig 1 byte]  payload           mode_id 1..0x3F, no checksum
  extended  [0xFF] version(varint) mode_id(varint) flags(1 byte) table_sig(2 bytes) [checksum] payload

The extended form exists because a one-byte mode field cannot carry a bigger id and because the optional
checksum (design decision, off by default) needs
somewhere to live: ``flags`` bits 0-1 = integrity code (0 none, 1 crc8, 2 crc16), checksum bytes
follow the signature and cover the payload. The default package never pays for either.

Why no clash with old packages: an old first byte is ``dict << 4 | variant`` with dict <= 4, so
always <= 0x4F. Every compact first byte is >= 0x81, extended is 0xFF. Anything else
(0x50..0x80, 0xC0..0xFE) is "not a hebzip package". A new version = a new ``version`` value plus
its own parser; the v1 parser never changes (unknown version -> ``UnsupportedFormat``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from hebzip.core.errors import CorruptPackage, UnsupportedFormat
from hebzip.domain.varint import decode_varint, encode_varint

COMPACT_FLAG: Final = 0x80
COMPACT_MAX_MODE: Final = 0x3F
EXTENDED_MAGIC: Final = 0xFF
LEGACY_MAX_BYTE: Final = 0x4F
FORMAT_VERSION: Final = 1
EXTENDED_SIG_BYTES: Final = 2
COMPACT_SIG_BYTES: Final = 1
INTEGRITY_BITS: Final = 0b11
CHECKSUM_SIZES: Final = {0: 0, 1: 1, 2: 2}
"""integrity code -> checksum bytes (must equal each ``IntegrityPort.size``; a test enforces it)."""


@dataclass(frozen=True)
class Header:
    """What a parsed header says. ``payload_at`` = offset of the first payload byte."""

    version: int
    mode_id: int
    table_sig: bytes
    integrity: int
    checksum: bytes
    payload_at: int


def kind_of(data: bytes) -> str:
    """``"v1"`` (ours), ``"legacy"`` (old raw package) or ``"unknown"`` - by the first byte only."""
    if not data:
        return "unknown"
    first = data[0]
    if first <= LEGACY_MAX_BYTE:
        return "legacy"
    if first == EXTENDED_MAGIC or COMPACT_FLAG < first <= COMPACT_FLAG | COMPACT_MAX_MODE:
        return "v1"
    return "unknown"


def build_header(mode_id: int, table_sig: bytes, integrity: int = 0, checksum: bytes = b"") -> bytes:
    """The header bytes. Compact when it can be (no checksum, small id), else extended."""
    if mode_id < 1:
        raise ValueError("mode id starts at 1")
    if len(checksum) != CHECKSUM_SIZES[integrity]:
        raise ValueError("checksum size does not fit the integrity code")
    if integrity == 0 and mode_id <= COMPACT_MAX_MODE:
        return bytes([COMPACT_FLAG | mode_id]) + table_sig[:COMPACT_SIG_BYTES]
    return (bytes([EXTENDED_MAGIC]) + encode_varint(FORMAT_VERSION) + encode_varint(mode_id)
            + bytes([integrity]) + table_sig[:EXTENDED_SIG_BYTES] + checksum)


def parse_header(data: bytes) -> Header:
    """Parse a v1 header; wrong magic / cut -> ``CorruptPackage``; unknown version -> ``UnsupportedFormat``."""
    if kind_of(data) != "v1":
        raise CorruptPackage("זו לא חבילת hebzip")
    first = data[0]
    if first != EXTENDED_MAGIC:
        return _compact(data, first)
    return _extended(data)


def _compact(data: bytes, first: int) -> Header:
    end = 1 + COMPACT_SIG_BYTES
    if len(data) < end:
        raise CorruptPackage("החבילה קטועה (כותרת)")
    return Header(FORMAT_VERSION, first & COMPACT_MAX_MODE, data[1:end], 0, b"", end)


def _extended(data: bytes) -> Header:
    version, pos = decode_varint(data, 1)
    if version != FORMAT_VERSION:
        raise UnsupportedFormat(f"גרסת כותרת {version} לא נתמכת (נתמכת: {FORMAT_VERSION})")
    mode_id, pos = decode_varint(data, pos)
    if pos >= len(data):
        raise CorruptPackage("החבילה קטועה (כותרת)")
    flags = data[pos]
    integrity = flags & INTEGRITY_BITS
    if flags & ~INTEGRITY_BITS or integrity not in CHECKSUM_SIZES:
        raise CorruptPackage("דגלי כותרת לא מוכרים - החבילה פגומה")
    sig_end = pos + 1 + EXTENDED_SIG_BYTES
    end = sig_end + CHECKSUM_SIZES[integrity]
    if len(data) < end or mode_id < 1:
        raise CorruptPackage("החבילה קטועה (כותרת)")
    return Header(version, mode_id, data[pos + 1:sig_end], integrity, data[sig_end:end], end)
