"""Header v1 (compact + extended) build/parse, and no clash with old packages."""

from __future__ import annotations

import pytest

from hebzip.core.errors import CorruptPackage, UnsupportedFormat
from hebzip.domain.header import CHECKSUM_SIZES, COMPACT_MAX_MODE, LEGACY_MAX_BYTE, build_header, kind_of, parse_header
from hebzip.domain.varint import encode_varint

SIG = bytes([0xAB, 0xCD])


def test_compact_is_two_bytes_and_round_trips() -> None:
    data = build_header(2, SIG) + b"PAYLOAD"
    assert data[:2] == bytes([0x82, 0xAB]) and kind_of(data) == "v1"
    h = parse_header(data)
    assert (h.version, h.mode_id, h.table_sig, h.integrity, h.payload_at) == (1, 2, bytes([0xAB]), 0, 2)
    assert data[h.payload_at:] == b"PAYLOAD"


def test_no_compact_first_byte_can_be_an_old_package_byte() -> None:
    old = {(d << 4) | v for d in range(5) for v in range(16)}
    assert max(old) <= LEGACY_MAX_BYTE
    compact = {0x80 | m for m in range(1, COMPACT_MAX_MODE + 1)}
    assert not compact & old and 0xFF not in old
    assert all(kind_of(bytes([b])) == "legacy" for b in old)


def test_big_mode_id_and_checksum_use_the_extended_form() -> None:
    big = build_header(1000, SIG)
    assert big[0] == 0xFF and parse_header(big).mode_id == 1000
    for code, size in CHECKSUM_SIZES.items():
        crc = bytes(range(size))
        h = parse_header(build_header(3, SIG, code, crc) + b"zz")
        assert (h.integrity, h.checksum, h.table_sig) == (code, crc, SIG if code else SIG[:1])  # code 0 = compact


def test_version_two_is_unsupported() -> None:
    data = bytes([0xFF]) + encode_varint(2) + encode_varint(1) + bytes([0]) + SIG
    with pytest.raises(UnsupportedFormat):
        parse_header(data)


def test_truncated_headers_are_corrupt() -> None:
    full = build_header(3, SIG, 2, b"\x01\x02")
    for cut in range(1, len(full)):
        with pytest.raises(CorruptPackage):
            parse_header(full[:cut])
    with pytest.raises(CorruptPackage):
        parse_header(bytes([0x83]))


def test_foreign_first_bytes_are_not_ours() -> None:
    for first in (0x50, 0x80, 0xB5 + 0x10, 0xC0, 0xFE):
        assert kind_of(bytes([first, 0, 0])) == "unknown"
        with pytest.raises(CorruptPackage):
            parse_header(bytes([first, 0, 0]))
    assert kind_of(b"") == "unknown"
