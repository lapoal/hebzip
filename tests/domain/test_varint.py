"""``encode_varint`` follows the package rule byte for byte (the golden payloads start with it)."""

from __future__ import annotations

import pytest

from hebzip.core.errors import CorruptPackage
from hebzip.domain.varint import decode_varint, encode_varint


def _rule(n: int) -> bytes:
    """The format rule written out: 7 bits per byte, least significant first, 0x80 = more follows."""
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def test_follows_the_rule_for_small_numbers() -> None:
    assert all(encode_varint(n) == _rule(n) for n in range(70001))


def test_follows_the_rule_around_powers_of_two() -> None:
    for k in range(41):
        for n in (2**k - 1, 2**k, 2**k + 1):
            assert encode_varint(n) == _rule(n)


def test_known_vectors() -> None:
    assert encode_varint(0) == bytes([0]) and encode_varint(127) == bytes([127])
    assert encode_varint(128) == bytes([0x80, 0x01]) and encode_varint(300) == bytes([0xAC, 0x02])


def test_decode_is_the_inverse() -> None:
    values = [0, 1, 127, 128, 300, 70000, 2**32, 2**40 + 1]
    for n in values:
        data = b"xx" + encode_varint(n) + b"yy"
        value, pos = decode_varint(data, 2)
        assert value == n and data[pos:] == b"yy"


def test_truncated_input_is_a_typed_error() -> None:
    with pytest.raises(CorruptPackage):
        decode_varint(bytes([0x80, 0x80]), 0)
    with pytest.raises(CorruptPackage):
        decode_varint(b"", 0)


def test_endless_varint_is_refused() -> None:
    with pytest.raises(CorruptPackage):
        decode_varint(bytes([0xFF] * 40), 0)
