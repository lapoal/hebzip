"""Conformance of the word-model drawers: base prior, each predictor, mixer, word model - real vs plate.

Every real drawer and its plate answer the same Protocol with the same signatures; every plate is
safe: the predictor plate is inactive (left out of the mixer), the base plate keeps every word
codable, the mixer plate never learns, the word-model plate refuses. Built through the real
assembly (python engine) where a drawer needs tables.
"""

from __future__ import annotations

import pytest

from hebzip.config.load import load_dict
from hebzip.core.errors import DrawerUnavailable
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import MID_GAP, BasePriorPort, MixerPort, PredictorPort, WordHistory, WordModelPort
from hebzip.domain.words.base import UniformBase
from hebzip.domain.words.mixer import StaticMixer, TrustMixer
from hebzip.domain.words.mixture import MixtureWordModel, NullWordModel
from hebzip.domain.words.predictors import NullPredictor
from hebzip.services.select import Assembly
from tests.support import ROOT, assert_same_signatures, python_config


def _fast() -> MixtureWordModel:
    model = Assembly(load_dict(python_config()), ROOT).parts("fast").words
    assert isinstance(model, MixtureWordModel)
    return model


@pytest.fixture(scope="module")
def words() -> MixtureWordModel:
    return _fast()


def test_base_real_and_plate(words: MixtureWordModel) -> None:
    real, plate = words.base, UniformBase(1000)
    for b in (real, plate):
        assert isinstance(b, BasePriorPort) and int(b.cum[-1]) == b.total and int(b.cum[0]) == 0
    assert_same_signatures(BasePriorPort, real, plate)
    assert plate.prob(7) == 1 / 1000 and real.prob(0) > real.prob(1_000_000) > 0.0


def test_every_predictor_and_the_plate(words: MixtureWordModel) -> None:
    plate = NullPredictor()
    h = WordHistory()
    for real in words.channels:
        assert isinstance(real, PredictorPort) and real.active
        assert_same_signatures(PredictorPort, real, plate)
        real.start()
        look = real.look(h)
        assert look.conf >= 0.0 and 0.0 <= look.escape <= 1.0
    assert isinstance(plate, PredictorPort) and not plate.active
    assert plate.look(h).conf == 0.0 and plate.sparse(1.0, 1 << 60) is None and plate.prob(5) == 0.0


def test_text_predictors_learn_and_corpus_backs_off() -> None:
    corpus, pairs, counts = _fast().channels
    h = WordHistory()
    for p in (corpus, pairs, counts):
        p.start()
    assert corpus.look(h).escape == 1.0 and corpus.sparse(1.0, 1 << 60) is None
    for x in (5, 9, 5, 9, 5):
        for p in (pairs, counts):
            p.look(h)
            p.observe(x, h)
        h.advance(x)
    h2 = WordHistory(prev=5)
    assert pairs.look(h2).conf > 0.0 and pairs.prob(9) == 1.0
    assert counts.look(h2).conf > 0.0 and counts.prob(5) == 3 / 5
    sparse = counts.sparse(0.5, 1 << 60)
    assert sparse is not None and list(sparse.ids) == [5, 9] and int(sparse.cum[0]) == 0


def test_mixer_real_and_plate() -> None:
    real, plate = TrustMixer(0.5, 0.1), StaticMixer()
    assert isinstance(real, MixerPort) and isinstance(plate, MixerPort)
    assert_same_signatures(MixerPort, real, plate)
    for m in (real, plate):
        m.start([1.0, 0.5])
        w = m.weights([1.0, 1.0])
        assert abs(w[0] + w[1] - 1.0) < 1e-12 and m.weights([0.0, 0.0]) == [0.0, 0.0]
        m.update(w, [0.9, 0.1])
        m.update(w, [0.0, 0.0])  # nothing predicted the word: nothing to learn
    assert plate.weights([1.0, 1.0])[0] == 2 / 3
    assert real.weights([1.0, 1.0])[0] > 2 / 3


def test_word_model_real_and_plate(words: MixtureWordModel) -> None:
    plate = NullWordModel()
    assert isinstance(words, WordModelPort) and isinstance(plate, WordModelPort)
    assert_same_signatures(WordModelPort, words, plate)
    enc = RangeEncoder(words.k)
    words.start()
    sent = [17, 42, 17, 1_000_000, 42]
    for x in sent:
        assert words.code(enc, x, MID_GAP) == x
    dec = RangeDecoder(words.k, enc.finish(), 1024)
    words.start()
    assert [words.code(dec, 0, MID_GAP) for _ in sent] == sent
    with pytest.raises(DrawerUnavailable):
        plate.code(enc, 3, MID_GAP)
