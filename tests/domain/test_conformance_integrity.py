"""Checksum drawer (design decision): CRC-8 / CRC-16 vs the null plate, and agreement with the header."""

from __future__ import annotations

from hebzip.domain.contracts import IntegrityPort
from hebzip.domain.header import CHECKSUM_SIZES
from hebzip.domain.integrity import Crc8Integrity, Crc16Integrity
from hebzip.domain.nulls import NullIntegrity
from tests.support import assert_same_signatures

CHECK = b"123456789"


def test_all_three_answer_the_contract() -> None:
    drawers = [NullIntegrity(), Crc8Integrity(), Crc16Integrity()]
    for drawer in drawers:
        assert isinstance(drawer, IntegrityPort)
    assert_same_signatures(IntegrityPort, *drawers)


def test_sizes_and_codes_agree_with_the_header_table() -> None:
    for drawer in (NullIntegrity(), Crc8Integrity(), Crc16Integrity()):
        assert CHECKSUM_SIZES[drawer.code] == drawer.size == len(drawer.compute(CHECK))


def test_known_check_values() -> None:
    assert Crc8Integrity().compute(CHECK) == bytes([0xF4])
    assert Crc16Integrity().compute(CHECK) == bytes([0x29, 0xB1])


def test_plate_is_silent_and_real_ones_notice_a_flipped_bit() -> None:
    assert NullIntegrity().compute(CHECK) == b""
    flipped = bytes([CHECK[0] ^ 1]) + CHECK[1:]
    for drawer in (Crc8Integrity(), Crc16Integrity()):
        assert drawer.compute(CHECK) != drawer.compute(flipped)
