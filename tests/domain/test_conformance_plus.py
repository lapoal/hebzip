"""Conformance of the max_plus drawers: each real drawer and its plate answer the same
Protocol with the same signatures, and the plate is safe (the conformance-test rule). Built through the real assembly
(python engine) from the real config and tables.

- ``corpus_classes`` vs the predictor plate ``null``: no history = no evidence; after a word its class
  list speaks (evidence 1, found, an escape in (0, 1]) and its probabilities sum with the escape to 1.
- ``trust_selected`` vs ``trust`` vs the plate ``static``: one Protocol; the situation key follows the
  gap class, the found flag and the position, and only the selected vector learns.
- ``ChoosePayload``: a choose mode never encodes or decodes a payload itself.
"""

from __future__ import annotations

import pytest

from hebzip.config.load import load_dict
from hebzip.core.errors import CorruptPackage, DrawerUnavailable
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import MID_GAP, Gap, Look, MixerPort, PayloadCodecPort, PredictorPort, WordHistory
from hebzip.domain.nulls import ChoosePayload
from hebzip.domain.words.mixer import StaticMixer, TrustMixer
from hebzip.domain.words.mixer_selected import SelectedTrustMixer
from hebzip.domain.words.mixture import MixtureWordModel
from hebzip.domain.words.predictors import NullPredictor
from hebzip.domain.words.predictors_classes import CorpusClasses
from hebzip.services.select import Assembly
from tests.support import ROOT, assert_same_signatures, python_config

MODE = "max_plus"


@pytest.fixture(scope="module")
def words() -> MixtureWordModel:
    built = Assembly(load_dict(python_config()), ROOT).parts(MODE)
    assert isinstance(built.words, MixtureWordModel)
    return built.words


def test_class_predictor_real_and_plate(words: MixtureWordModel) -> None:
    real = [p for p in words.channels if isinstance(p, CorpusClasses)]
    assert [p.window for p in real] == [1, 2]
    plate = NullPredictor()
    for p in real:
        assert isinstance(p, PredictorPort) and p.active
        assert_same_signatures(PredictorPort, p, plate)
        p.start()
        assert p.look(WordHistory()) == Look(0.0, 0.0) and p.sparse(1.0, 1 << 60) is None and p.prob(5) == 0.0
    h = WordHistory(prev=17, prev2=42, n=2)
    for p in real:
        look = p.look(h)
        assert look.conf == 1.0 and look.found and 0.0 < look.escape <= 1.0
        part = p.sparse(1.0, 1 << 40)
        assert part is not None and len(part.ids) > 0 and bool((part.ids[1:] > part.ids[:-1]).all())
        total = look.escape
        for x in part.ids.tolist():
            total += p.prob(int(x))
        assert abs(total - 1.0) < 1e-9, "the listed followers plus the escape are the whole mass"
    assert real[0].look(WordHistory(prev=17)).found and not real[1].look(WordHistory(prev=17)).found, (
        "window 2 needs the word two back")


def test_selected_mixer_real_trust_and_plate(words: MixtureWordModel) -> None:
    real = words.mixer
    assert isinstance(real, SelectedTrustMixer)
    trust, plate = TrustMixer(0.05, 0.003), StaticMixer()
    for m in (real, trust, plate):
        assert isinstance(m, MixerPort)
    assert_same_signatures(MixerPort, real, trust, plate)
    n = len(words.channels)
    real.start([1.0] * n)
    found = [Look(1.0, 0.5, found=True)] + [Look(0.5, 0.0)] * (n - 1)
    lost = [Look(1.0, 1.0)] + [Look(0.5, 0.0)] * (n - 1)
    keys = []
    for h, looks in ((WordHistory(n=0, gap=MID_GAP), found), (WordHistory(n=0, gap=MID_GAP), lost),
                     (WordHistory(n=12, gap=MID_GAP), found), (WordHistory(n=12, gap=Gap(1, 1)), found)):
        real.select(h, looks)
        keys.append(real.key)
    assert keys == [2, 0, 3, 7], "key = (gap class * 2 + found) * 2 + (n >= position_split)"
    w = real.weights([lk.conf for lk in found])
    assert abs(sum(w) - 1.0) < 1e-12
    before = {k: list(v) for k, v in real.live.items()}
    real.update(w, [0.9] + [0.01] * (n - 1))
    assert real.live[7] != before[7] and all(real.live[k] == before[k] for k in before if k != 7), (
        "only the selected vector learns")
    plate.start([1.0] * n)
    plate.update(w, [0.9] * n)
    assert plate.weights([1.0] * n) == [1.0 / n] * n, "the plate never learns"


def test_choose_payload_refuses_both_ways() -> None:
    plate = ChoosePayload()
    assert isinstance(plate, PayloadCodecPort)
    with pytest.raises(DrawerUnavailable):
        plate.encode("שלום")
    with pytest.raises(CorruptPackage):
        plate.decode(b"\x00", 100)


def test_a_plated_channel_drops_its_pretrained_trust() -> None:
    """The trust table has one trust per CONFIGURED channel; the mixer sees only the active ones, so a channel
    switched to its plate must take its trust with it - else every vector is one too long."""
    cfg = python_config()
    cfg["modes"][MODE]["predictors"]["class_pairs"]["kind"] = "null"
    words = Assembly(load_dict(cfg), ROOT).parts(MODE).words
    assert isinstance(words, MixtureWordModel) and isinstance(words.mixer, SelectedTrustMixer)
    n = len(words.channels)
    assert n == len(cfg["modes"][MODE]["channels"]) - 1
    assert all(len(v) == n for v in words.mixer.sets.values())
    enc = RangeEncoder(words.k)
    words.start()
    sent = [17, 42, 17, 5, 42]
    assert [words.code(enc, x, MID_GAP) for x in sent] == sent
    dec = RangeDecoder(words.k, enc.finish(), 1024)
    words.start()
    assert [words.code(dec, 0, MID_GAP) for _ in sent] == sent
