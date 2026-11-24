"""The selector: its words are exactly the config's vocabulary, a missing row means
the plate plus a log line, and the real fast mode is built from real drawers and signed."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from hebzip.config import schema as S
from hebzip.config.load import load_dict
from hebzip.config.schema import KIND_ROWS, MODE_WORDS
from hebzip.config.validate_model import MODE_KIND_CHECKS
from hebzip.core.errors import ConfigError
from hebzip.domain.contracts import WordHistory
from hebzip.domain.spell.speller import SpelledWordCoder
from hebzip.domain.words.mixture import MixtureWordModel
from hebzip.services.select import REAL, Assembly
from tests.support import ROOT, real_config

FAMILY_OF_KIND = {"template": "template", "base": "base", "mixer": "mixer", "predictor": "predictor", "gap": "gap",
                  "letter": "letter", "letter_mixer": "letter_mixer"}


def _python_config() -> dict[str, Any]:
    cfg = real_config()
    cfg["hebzip"]["engine"] = "python"
    return cfg


def test_selector_words_are_the_config_vocabulary() -> None:
    for drawer, words in MODE_WORDS.items():
        assert set(REAL[drawer]) == set(words), drawer
    for family in FAMILY_OF_KIND.values():
        assert set(REAL[family]) == set(KIND_ROWS[family]), family
    assert set(REAL["integrity"]) == {"none", "crc8", "crc16"}
    # the engines, the generic codec families and the fallback policies are selector tables too
    assert set(REAL["engine"]) == set(S.ENGINE_WORDS)
    assert set(REAL["codec"]) == set(S.CODEC_FORMS)
    assert set(REAL["fallback_policy"]) == set(S.FALLBACK_POLICIES)
    assert set(REAL["table_store"]) == set(S.STORE_WORDS)
    assert set(MODE_KIND_CHECKS) == set(S.KINDS) == set(S.FALLBACK_KINDS) == set(REAL["payload"])


def test_missing_rows_become_plates_and_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _python_config()
    fast = cfg["modes"]["fast"]
    for key in ("dictionary", "tokenizer", "word_model", "template", "word_coder", "gap"):
        del fast[key]
    asm = Assembly(load_dict(cfg), ROOT)
    with caplog.at_level(logging.WARNING, logger="hebzip"):
        parts = asm.parts("fast")
    assert (parts.dictionary.name, parts.tokenizer.name, parts.words.name, parts.template.name) == (
        "null", "literal", "null", "flat")
    defaulted = [r for r in caplog.records if r.getMessage() == "drawer_defaulted_to_plate"]
    assert len(defaulted) == 5  # dictionary, tokenizer, template, gap, word model (its word coder is never asked)
    assert parts.gap.name == "none"


def test_assembly_builds_the_real_fast_mode_and_signs_it() -> None:
    asm = Assembly(load_dict(_python_config()), ROOT)
    parts = asm.parts("fast")
    assert (parts.dictionary.name, parts.tokenizer.name, parts.words.name, parts.template.name) == (
        "word_list", "overlay", "mixture", "cm")
    assert isinstance(parts.words, MixtureWordModel)
    assert [p.name for p in parts.words.channels] == ["corpus_pairs", "text_pairs", "text_counts"]
    assert (parts.words.base.name, parts.words.mixer.name) == ("unigram_zipf", "trust")
    assert parts.kernels.engine == "python" and parts.phantom_max == 1024
    assert len(asm.signature("fast")) == 32 and asm.signature("fast") != asm.signature("max")
    assert asm.generic("generic").name == "best_of"
    assert (asm.mode_by_id(2), asm.mode_by_id(99)) == ("max", None)
    assert asm.integrity("none").size == 0 and asm.integrity_by_code(2).name == "crc16"
    assert WordHistory().prev is None


def test_a_switched_off_predictor_is_left_out_of_the_mixer() -> None:
    cfg = _python_config()
    cfg["modes"]["fast"]["predictors"]["text_pairs"]["kind"] = "null"
    parts = Assembly(load_dict(cfg), ROOT).parts("fast")
    assert isinstance(parts.words, MixtureWordModel)
    assert [p.name for p in parts.words.channels] == ["corpus_pairs", "text_counts"]


@pytest.mark.parametrize(("path", "value"), [(("modes", "max", "gap", "groups"), [0, 1, 2]),
                                             (("languages", "he", "gap_marks"), [",;"])])
def test_gap_groups_must_fit_the_classifier_at_start_up(path: tuple[str, ...], value: Any) -> None:
    """The class count comes from the classifier itself (one source), checked when the Assembly starts."""
    cfg = _python_config()
    node: Any = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError, match=r"\[modes\.max\.gap\]\.groups"):
        Assembly(load_dict(cfg), ROOT)


def test_assembly_builds_the_real_max_mode() -> None:
    parts = Assembly(load_dict(_python_config()), ROOT).parts("max")
    assert isinstance(parts.words, MixtureWordModel) and parts.gap.name == "punctuation"
    assert [p.name for p in parts.words.channels] == ["corpus_pairs", "text_pairs", "text_counts", "text_triples",
                                                      "text_gap_starts", "corpus_gap_starts", "prefix_class", "recency"]
    coder = parts.words.word_coder
    assert isinstance(coder, SpelledWordCoder)
    assert [i.name for i in coder.inputs] == ["static_chain", "static_chain", "text_letters"]
    assert coder.mixer.name == "geometric" and coder.history == 6


def test_more_adaptive_inputs_than_the_key_layout_holds_is_refused_at_start() -> None:
    """The adaptive key gives the input number 4 bits; a 17th input used to share a key
    silently. 16 adaptive orders + the word context = 17 inputs -> refused, naming the row; 16 fit."""
    cfg = _python_config()
    cfg["modes"]["fast"]["template"]["adaptive"] = list(range(16))
    with pytest.raises(ConfigError, match=r"\[modes\.fast\.template\]\.adaptive.*at most 16 adaptive inputs"):
        Assembly(load_dict(cfg), ROOT)
    cfg["modes"]["fast"]["template"]["adaptive"] = list(range(15))
    Assembly(load_dict(cfg), ROOT)
