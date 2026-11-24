"""Damaged packages end in bounded work with a typed error (design decision-1).

The defect: past the end of the body the decoder reads zero bits forever, so a package that declares
a long template (any length up to ``max_output_bytes``) kept the decoder busy for hours. The fix is
the decoder's phantom-bit budget (``phantom_bits_max``). Through the real CLI in a child process, for
mode fast and for mode max (the spelled word coder checks the same budget after every letter).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest

from hebzip.domain.varint import encode_varint
from tests.support import compress_file, decompress_file, sample_paths, work_dir

FAKE_TEMPLATE = 150_000_000
BOUND_S = 120
TYPED_EXITS = {3, 4}
MODES = ("fast", "max")


def _package(work: Path, text: Path, mode: str) -> bytes:
    pkg = work / f"{text.stem}.hz"
    done = compress_file(text, pkg, mode, "--no-fallback")
    assert done.returncode == 0, done.stderr[-300:]
    return pkg.read_bytes()


@pytest.mark.parametrize("mode", MODES)
def test_fake_template_length_stops_with_a_typed_error(mode: str) -> None:
    work = work_dir(f"damaged_fake_length_{mode}")
    real = _package(work, sample_paths()[0], mode)
    forged = real[:2] + encode_varint(FAKE_TEMPLATE) + bytes([0x5A, 0xC3, 0x01])
    src = work / "forged.hz"
    src.write_bytes(forged)
    started = time.monotonic()
    done = decompress_file(src, work / "forged.out", timeout=BOUND_S)
    elapsed = time.monotonic() - started
    assert done.returncode == 4, (done.returncode, done.stderr[-400:])
    assert "phantom_bits_max" in done.stderr.decode("utf-8", "replace")
    assert not (work / "forged.out").exists()
    assert elapsed < BOUND_S


def _damaged(data: bytes, rnd: random.Random) -> list[bytes]:
    body = len(data) - 2
    flip = bytearray(data)
    flip[2 + rnd.randrange(body)] ^= 1 << rnd.randrange(8)
    return [bytes(flip), data[:2 + body // 2], data + bytes(rnd.getrandbits(8) for _ in range(9)),
            data[:1] + bytes([data[1] ^ 0x40]) + data[2:]]


@pytest.mark.parametrize("mode", MODES)
def test_random_damage_is_text_or_a_typed_error_never_a_crash(mode: str) -> None:
    work = work_dir(f"damaged_random_{mode}")
    rnd = random.Random(7)
    for text in sample_paths():
        for i, bad in enumerate(_damaged(_package(work, text, mode), rnd)):
            src = work / f"{text.stem}_{i}.hz"
            src.write_bytes(bad)
            done = decompress_file(src, work / f"{text.stem}_{i}.out", timeout=BOUND_S)
            assert done.returncode == 0 or done.returncode in TYPED_EXITS, (text.stem, i, done.stderr[-400:])
