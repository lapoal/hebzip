"""Conformance of mode max's drawers: gap classifier, the five new predictors, the word coders, the
letter inputs and the letter mixer - each real drawer and its plate answer the same Protocol with the
same signatures, and every plate is safe (the conformance-test rule). Built through the real assembly
(python engine) from the real config and tables.
"""

from __future__ import annotations

from math import exp

import pytest

from hebzip.config.load import load_dict
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import (
    MID_GAP,
    Gap,
    GapClassifierPort,
    LetterInputPort,
    LetterMixerPort,
    PredictorPort,
    WordCoderPort,
    WordHistory,
)
from hebzip.domain.gap import NoGap, PunctuationGap
from hebzip.domain.spell.inputs import NullLetterInput
from hebzip.domain.spell.mixer import GeometricMixer, WordOnlyMixer
from hebzip.domain.spell.speller import SpelledWordCoder
from hebzip.domain.words.coders import DirectWordCoder
from hebzip.domain.words.mixture import MixtureWordModel
from hebzip.domain.words.predictors import NullPredictor
from hebzip.services.select import Assembly
from tests.support import ROOT, assert_same_signatures, python_config

NEW = ("text_triples", "text_gap_starts", "corpus_gap_starts", "prefix_class", "recency")
SENT, AFTER_MARK = 1, 2


@pytest.fixture(scope="module")
def parts() -> tuple[MixtureWordModel, GapClassifierPort]:
    built = Assembly(load_dict(python_config()), ROOT).parts("max")
    assert isinstance(built.words, MixtureWordModel)
    return built.words, built.gap


def test_gap_real_and_plate(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    real, plate = parts[1], NoGap()
    assert isinstance(real, PunctuationGap) and isinstance(plate, GapClassifierPort)
    assert_same_signatures(GapClassifierPort, real, plate)
    cases: dict[tuple[bytes | str, bool], Gap] = {
             (b"", True): Gap(1, SENT), (b" ", False): MID_GAP, (b"", False): MID_GAP, (". ", False): Gap(1, SENT),
             (", ", False): Gap(2, AFTER_MARK), (": ", False): Gap(3, AFTER_MARK), (' "', False): Gap(4, AFTER_MARK),
             (" (", False): Gap(5, AFTER_MARK), (" - ", False): Gap(6, AFTER_MARK), (" 7 ", False): Gap(7, AFTER_MARK),
             (b"\xff", False): MID_GAP, ("\n", False): Gap(1, SENT)}
    for (gap, first), want in cases.items():
        raw = gap if isinstance(gap, bytes) else gap.encode("utf-8")
        assert real.classify(raw, first) == want, (gap, first)
        assert plate.classify(raw, first) == MID_GAP
    assert (real.classes, plate.classes, plate.groups) == (8, 1, 1)


def test_new_predictors_and_the_plate(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    channels = {p.name: p for p in parts[0].channels}
    plate = NullPredictor()
    for name in NEW:
        real = channels[name]
        assert isinstance(real, PredictorPort) and real.active
        assert_same_signatures(PredictorPort, real, plate)
        real.start()
        assert real.look(WordHistory()).conf == 0.0 and real.sparse(1.0, 1 << 60) is None


def _feed(p: PredictorPort, words: list[int], gap: Gap = MID_GAP) -> WordHistory:
    h = WordHistory(gap=gap)
    p.start()
    for x in words:
        p.look(h)
        p.observe(x, h)
        h.advance(x)
    return h


def test_text_predictors_learn_what_they_say(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    ch = {p.name: p for p in parts[0].channels}
    tri = ch["text_triples"]
    h = _feed(tri, [5, 9, 7, 5, 9])
    assert tri.look(h).conf > 0.0 and tri.prob(7) == 1.0
    tg = ch["text_gap_starts"]
    _feed(tg, [11, 12], Gap(1, SENT))
    assert tg.look(WordHistory(gap=Gap(1, SENT))).conf > 0.0 and tg.prob(12) == 0.5
    assert tg.look(WordHistory(gap=Gap(0, 0))).conf == 0.0, "a group that is not switched on is silent"
    rc = ch["recency"]
    h = _feed(rc, list(range(40)))
    rc.look(h)
    assert rc.prob(39) == 1 / 30 and rc.prob(0) == 0.0, "the window keeps the last 30 words"


def test_corpus_predictors_back_off_and_classify(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    ch = {p.name: p for p in parts[0].channels}
    cg = ch["corpus_gap_starts"]
    cg.start()
    look = cg.look(WordHistory(gap=Gap(1, SENT)))
    assert look.conf == 1.0 and 0.0 < look.escape < 1.0 and cg.sparse(0.5, 1 << 60) is not None
    assert cg.look(WordHistory(gap=MID_GAP)).conf == 0.0
    px = ch["prefix_class"]
    h = _feed(px, [17, 42, 17])
    assert px.look(h).conf > 0.0 and px.prob(17) > 0.0 and px.prob(42) > 0.0 and px.prob(999_999) == 0.0


def test_word_coders_real_and_plate(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    words = parts[0]
    real, plate = words.word_coder, DirectWordCoder()
    assert isinstance(real, SpelledWordCoder) and isinstance(plate, WordCoderPort)
    assert_same_signatures(WordCoderPort, real, plate)
    sent = [(17, Gap(1, SENT)), (42, MID_GAP), (17, Gap(2, AFTER_MARK)), (1_000_000, MID_GAP), (42, MID_GAP)]
    for coder in (real, plate):
        words.word_coder = coder
        enc = RangeEncoder(words.k)
        words.start()
        assert [words.code(enc, x, g) for x, g in sent] == [x for x, _ in sent]
        dec = RangeDecoder(words.k, enc.finish(), 1024)
        words.start()
        assert [words.code(dec, 0, g) for _, g in sent] == [x for x, _ in sent]
    words.word_coder = real


def test_letter_inputs_real_and_plate(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    speller = parts[0].word_coder
    assert isinstance(speller, SpelledWordCoder)
    plate = NullLetterInput()
    assert isinstance(plate, LetterInputPort) and not plate.active and plate.reach == 0
    hist = [-1] * speller.history + [3]
    codes = list(range(speller.index.symbols))
    for real in speller.inputs:
        assert isinstance(real, LetterInputPort) and real.active
        assert_same_signatures(LetterInputPort, real, plate)
        real.start()
        logs = real.logs(hist, codes)
        total = 0.0
        for v in logs:
            total += exp(v)
        assert len(logs) == len(codes) and abs(total - 1.0) < 1e-9
    text = speller.inputs[-1]
    before = text.logs(hist, [5])[0]
    for _ in range(3):
        text.observe(hist, 5)
    assert text.logs(hist, [5])[0] > before, "the text's own letters raise what it saw"


def test_letter_mixer_real_and_plate(parts: tuple[MixtureWordModel, GapClassifierPort]) -> None:
    speller = parts[0].word_coder
    assert isinstance(speller, SpelledWordCoder) and isinstance(speller.mixer, GeometricMixer)
    real, plate = speller.mixer, WordOnlyMixer(3)
    assert isinstance(plate, LetterMixerPort)
    assert_same_signatures(LetterMixerPort, real, plate)
    masses = [600, 300, 100]
    ins: list[list[float] | None] = [[-0.5108256237659907, -1.2039728043259361, -2.3025850929940455],
                                     [-2.0, -0.2, -3.0], None, [-1.0, -1.0, -1.5]]
    for m in (real, plate):
        m.start()
        w = m.select(2, masses, 1000)
        before = list(w)
        p = m.probs(w, ins)
        m.learn(w, ins, p, 1)
        assert abs(p[0] + p[1] + p[2] - 1.0) < 1e-12 and w[2] == before[2], "an input that is not mixed stays"
        assert (w != before) == (m is real), "only the real mixer learns"
    p = plate.probs(plate.select(0, masses, 1000), ins)
    assert all(abs(a - b / 1000) < 1e-12 for a, b in zip(p, masses, strict=True)), "the plate = the word model"
