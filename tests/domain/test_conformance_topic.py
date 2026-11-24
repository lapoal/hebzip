"""Conformance of the topic channel ``text_topic`` and its plate ``null`` (the conformance-test rule).

- one Protocol, same signatures; nothing seen = no evidence, no part, probability 0 (the plate's answers);
- a word WITHOUT a neighbour list (frequent / very rare) brings no topic; a content word brings exactly its list;
- the window: after ``window`` + 1 content words the oldest one's list is gone - the counts are always exactly the
  sum of the lists of the last ``window`` content words (integers, recomputed here from the table);
- the probabilities of the listed ids sum to 1 (escape 0) and the sparse part is the text-table step;
- built from the REAL config and table (mode ``max_plus``), every table answer is in the dictionary;
- the whole word model with the topic channel codes and decodes the same ids (both sides compute the same numbers).
"""

from __future__ import annotations

import numpy as np
import pytest

from hebzip.config.load import load_dict
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import MID_GAP, Look, PredictorPort, WordHistory
from hebzip.domain.words.mixture import MixtureWordModel
from hebzip.domain.words.predictors import NullPredictor
from hebzip.domain.words.predictors_topic import NeighbourLists, TextTopic
from hebzip.services.select import Assembly
from tests.support import ROOT, assert_same_signatures, python_config

MODE = "max_plus"
# word 0, 1: no list (function words); 2 -> {3: 5, 4: 1}; 3 -> {2: 7}; 4 -> {2: 2, 5: 9}; 5: no list
OFF = np.array([0, 0, 0, 2, 3, 5, 5], dtype=np.int64)
IDS = np.array([3, 4, 2, 2, 5], dtype=np.int32)
WT = np.array([5, 1, 7, 2, 9], dtype=np.uint16)


def _toy(window: int = 2) -> TextTopic:
    return TextTopic(NeighbourLists(OFF, IDS, WT), window, 4.0, 0.1)


def _expected(content: list[int], window: int) -> dict[int, int]:
    acc: dict[int, int] = {}
    for y in content[-window:]:
        for x, w in zip(IDS[OFF[y]:OFF[y + 1]].tolist(), WT[OFF[y]:OFF[y + 1]].tolist(), strict=True):
            acc[x] = acc.get(x, 0) + w
    return {x: c for x, c in acc.items() if c}


def test_topic_real_and_plate_answer_the_same_contract() -> None:
    real, plate = _toy(), NullPredictor()
    assert isinstance(real, PredictorPort) and real.active and not plate.active
    assert_same_signatures(PredictorPort, real, plate)
    for p in (real, plate):
        p.start()
        assert p.look(WordHistory()) == Look(0.0, 0.0) and p.sparse(1.0, 1 << 60) is None and p.prob(3) == 0.0


def test_only_content_words_bring_a_topic_and_the_window_forgets_the_oldest() -> None:
    p = _toy(window=2)
    h = WordHistory()
    for x in (0, 1, 5):
        p.observe(x, h)
    assert p.look(h) == Look(0.0, 0.0) and p.acc == {}, "words without a list bring nothing"
    seen: list[int] = []
    for x in (2, 0, 3, 4, 2):
        p.observe(x, h)
        seen += [x] if x in (2, 3, 4) else []
        assert p.acc == _expected(seen, 2) and p.total == sum(p.acc.values())
        assert p.look(h).conf == len(seen[-2:]) / (len(seen[-2:]) + 4.0) and p.look(h).escape == 0.0
    part = p.sparse(1.0, 1 << 40)
    assert part is not None and part.ids.tolist() == sorted(p.acc)
    assert abs(sum(p.prob(int(x)) for x in part.ids) - 1.0) < 1e-12, "escape 0: the listed ids are the whole mass"
    f = (1 << 40) / p.total
    assert np.diff(part.cum).tolist() == [int(p.acc[x] * f) for x in sorted(p.acc)], "the text-table step"
    p.start()
    assert p.acc == {} and p.total == 0 and p.look(h) == Look(0.0, 0.0)


@pytest.fixture(scope="module")
def words() -> MixtureWordModel:
    built = Assembly(load_dict(python_config()), ROOT).parts(MODE)
    assert isinstance(built.words, MixtureWordModel)
    return built.words


def test_the_real_table_gives_dictionary_words_and_the_model_round_trips(words: MixtureWordModel) -> None:
    topic = [p for p in words.channels if isinstance(p, TextTopic)]
    assert len(topic) == 1 and topic[0] is words.channels[-1], "the topic channel is the last configured channel"
    t = topic[0]
    off = t.lists.off
    with_list = np.flatnonzero(np.diff(off) > 0)
    assert len(with_list) > 1000
    y = int(with_list[len(with_list) // 2])
    ids, wts = t.lists.at(y)
    assert ids == sorted(ids) and y not in ids and min(wts) >= 1 and max(ids) < len(off) - 1
    sent = [int(v) for v in with_list[[10, 500, 2000, 10, 3000, 500]]] + [0, 1, 2]
    enc = RangeEncoder(words.k)
    words.start()
    assert [words.code(enc, x, MID_GAP) for x in sent] == sent
    dec = RangeDecoder(words.k, enc.finish(), 1024)
    words.start()
    assert [words.code(dec, 0, MID_GAP) for _ in sent] == sent
