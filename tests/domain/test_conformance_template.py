"""Conformance of the template-model drawers (``cm`` and the plate ``flat``) and of the coder ends.

Both answer the same contract; both round-trip arbitrary bytes on one coder; the plate costs exactly
one bit per bit (safe: always works, learns nothing). Built through the real assembly, python engine.
"""

from __future__ import annotations

import random

from hebzip.config.load import load_dict
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import CoderPort, TemplateModelPort
from hebzip.domain.template_cm import CmTemplateModel, FlatTemplateModel
from hebzip.services.assemble import Loader, cm_template
from hebzip.services.select import engine_kernels, table_store
from tests.support import ROOT, assert_same_signatures, python_config

SETTINGS = load_dict(python_config())
KERNELS = engine_kernels("python", ROOT)


def _models() -> tuple[CmTemplateModel, FlatTemplateModel]:
    row = SETTINGS.modes["fast"].template
    assert row is not None
    return cm_template(row, Loader(table_store(SETTINGS, ROOT)), KERNELS), FlatTemplateModel(KERNELS)


def _round(model: TemplateModelPort, data: bytes) -> tuple[bytes, bytes]:
    enc = RangeEncoder(KERNELS)
    model.start(len(data))
    for i, b in enumerate(data):
        model.code_byte(enc, b)
        if i % 7 == 6:
            model.set_word(i)
    body = enc.finish()
    dec = RangeDecoder(KERNELS, body, 1024)
    model.start(len(data))
    out = bytearray()
    for i in range(len(data)):
        out.append(model.code_byte(dec, 0))
        if i % 7 == 6:
            model.set_word(i)
    return body, bytes(out)


def test_real_and_plate_share_the_contract() -> None:
    real, plate = _models()
    assert isinstance(real, TemplateModelPort) and isinstance(plate, TemplateModelPort)
    assert_same_signatures(TemplateModelPort, real, plate)
    assert isinstance(RangeEncoder(KERNELS), CoderPort) and isinstance(RangeDecoder(KERNELS, b"", 1024), CoderPort)


def test_both_round_trip_and_the_plate_costs_one_bit_per_bit() -> None:
    real, plate = _models()
    rnd = random.Random(3)
    data = "שלום, עולם! hello 123 ".encode() * 5 + bytes(rnd.getrandbits(8) for _ in range(64))
    body_real, back_real = _round(real, data)
    body_flat, back_flat = _round(plate, data)
    assert back_real == data and back_flat == data
    assert len(body_flat) in (len(data), len(data) + 1)  # 8 bits per byte (+ the final settle byte)
    assert len(body_real) < len(body_flat)


def test_the_notebook_grows_without_changing_results() -> None:
    real, _ = _models()
    text = bytes(random.Random(5).getrandbits(8) for _ in range(3000))
    body, back = _round(real, text)
    assert back == text and real.ad[0].shape[0] > 1 << 12
