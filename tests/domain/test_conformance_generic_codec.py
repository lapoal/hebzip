"""Conformance of the fallback drawer (``best_of`` vs the plate ``null``) - frames equal the golden set.

The codec list is a config row; with the research prototype's 15 codecs in its order every frame is
byte-identical to the payloads of ``golden/generic``. Damaged frames are typed errors; the plate refuses.
"""

from __future__ import annotations

import pytest

from hebzip.config.load import load_config
from hebzip.core.errors import CorruptPackage, DrawerUnavailable
from hebzip.domain.contracts import GenericCodecPort
from hebzip.domain.nulls import NullGenericCodec
from hebzip.integrations.generic import BestOfCodec
from hebzip.services.select import REAL
from tests.support import CONFIG, assert_same_signatures, golden_package, text_paths


def _real() -> BestOfCodec:
    """The real drawer exactly as the selector builds it (each codec word through the codec registry)."""
    row = load_config(CONFIG).modes["generic"]
    codec: BestOfCodec = REAL["generic_codec"]["best_of"](row)
    return codec


def test_real_and_plate_share_the_contract() -> None:
    real, plate = _real(), NullGenericCodec()
    assert isinstance(real, GenericCodecPort) and isinstance(plate, GenericCodecPort)
    assert_same_signatures(GenericCodecPort, real, plate)
    assert len(real.codec_names()) == 15 and plate.codec_names() == []
    with pytest.raises(DrawerUnavailable):
        plate.encode(b"x")


def test_frames_equal_the_golden_generic_set_and_decode() -> None:
    real = _real()
    for path in text_paths():
        raw = path.read_bytes()
        frame = real.encode(raw)
        assert frame == golden_package("generic", path.stem)[2:], path.stem
        assert real.decode(frame) == raw


def test_damaged_frames_are_typed_errors() -> None:
    real = _real()
    frame = real.encode("שלום עולם, hello world ".encode() * 20)
    for bad in (b"", bytes([99, 1, 2]), frame[:len(frame) // 2], frame[:1] + b"\x05" + frame[2:]):
        with pytest.raises(CorruptPackage):
            real.decode(bad)
