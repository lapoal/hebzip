"""Variable-length integers, byte-identical to the research prototype's ``ec.vint`` / ``ec.rv``.

Why our own copy and not ``ec``: the header (domain, zero I/O) must not import research prototype code.
Why it is safe: ``test_varint`` compares this file with ``ec.vint`` for 0..70000 and 2^k +-1 up to
2^40, so the payload's leading ``vint(len(template))`` stays the old one. 7 bits per byte,
little-endian, continuation bit 0x80.
"""

from __future__ import annotations

from hebzip.core.errors import CorruptPackage

MAX_SHIFT = 63
"""A length needs far fewer than 10 bytes; a longer run is damage, not a number."""


def encode_varint(n: int) -> bytes:
    """``n >= 0`` as a varint."""
    if n < 0:
        raise ValueError("varint is for non-negative numbers")
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Read a varint at ``pos`` -> ``(value, next_pos)``; cut or endless -> ``CorruptPackage``."""
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise CorruptPackage("החבילה קטועה (מספר באמצע הכותרת)")
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > MAX_SHIFT:
            raise CorruptPackage("מספר ארוך מדי בכותרת - החבילה פגומה")
