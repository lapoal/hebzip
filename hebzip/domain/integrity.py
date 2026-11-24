"""Optional payload checksums - the real ``IntegrityPort`` drawers (a design decision).

Why a drawer and not a switch in the compressor: the format has no checksum (neither did the old
one), so a flipped bit can decode to wrong text without any error. It is OFF by
default (it costs 1-2 bytes) and ON by one config word (``integrity = "crc16"``). The null plate
(``domain/nulls.NullIntegrity``) is the default. Pure math, zero I/O, hence in ``domain``.
CRC-8: poly 0x07, init 0. CRC-16: CCITT-FALSE (poly 0x1021, init 0xFFFF) via ``binascii.crc_hqx``.
"""

from __future__ import annotations

import binascii

CRC8_POLY = 0x07
CRC16_INIT = 0xFFFF


def _crc8_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


CRC8_TABLE = _crc8_table()


class Crc8Integrity:
    """One checksum byte (catches any single flipped bit and most bursts)."""

    name = "crc8"
    code = 1
    size = 1

    def compute(self, payload: bytes) -> bytes:
        """CRC-8 of ``payload`` as 1 byte."""
        crc = 0
        for byte in payload:
            crc = CRC8_TABLE[crc ^ byte]
        return bytes([crc])


class Crc16Integrity:
    """Two checksum bytes (1 in 65,536 corrupt payloads slips through)."""

    name = "crc16"
    code = 2
    size = 2

    def compute(self, payload: bytes) -> bytes:
        """CRC-16/CCITT-FALSE of ``payload`` as 2 big-endian bytes."""
        return binascii.crc_hqx(payload, CRC16_INIT).to_bytes(2, "big")
