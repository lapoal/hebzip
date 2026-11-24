"""Config validation of mode and drawer rows: a drawer word demands exactly its rows,
every number has a range, dependencies are refused at start-up, a plate is one word."""

from __future__ import annotations

from typing import Any

import pytest

from hebzip.config.load import load_dict
from tests.config.test_load import Path, refused, set_path
from tests.support import real_config

F, T, B, M, P, G = (("modes", "fast"), ("modes", "fast", "template"), ("modes", "fast", "base"),
                    ("modes", "fast", "mixer"), ("modes", "fast", "predictors"), ("modes", "generic"))
X, XP, XC, XL = (("modes", "max"), ("modes", "max", "predictors"), ("modes", "max", "letter_chains"),
                 ("modes", "max", "letters"))
BAD: list[tuple[Path, Any, str]] = [
    ((*F, "kind"), "other", "[modes.fast].kind"),
    ((*F, "id"), 0, "[modes.fast].id"),
    (("modes", "max", "id"), 1, "[modes.max].id"),
    ((*F, "retired"), "no", "retired"),
    ((*F, "word_model"), "letters", "[modes.fast].word_model"),
    ((*F, "dictionary"), "hc", "[modes.fast].dictionary"),
    ((*F, "dictionary"), "null", "word_model = 'mixture' needs dictionary"),
    ((*F, "tokenizer"), "regex", "[modes.fast].tokenizer"),
    ((*F, "generic_codec"), "best_of", "[modes.fast].generic_codec"),
    ((*F, "table_scale_bits"), 62, "[modes.fast].table_scale_bits"),
    ((*F, "channels"), ["corpus_pairs"], "[modes.fast].channels"),
    ((*F, "channels"), ["corpus_pairs", "text_pairs", "text_pairs"], "[modes.fast].channels"),
    ((*T, "kind"), "tiny", "[modes.fast.template].kind"),
    ((*T, "table"), "ghost", "[modes.fast.template].table"),
    ((*T, "orders"), [], "[modes.fast.template].orders"),
    ((*T, "orders"), [1, 1], "[modes.fast.template].orders"),
    ((*T, "adaptive"), [40], "[modes.fast.template].adaptive"),
    ((*T, "squash_knots"), [1, 2], "[modes.fast.template].squash_knots"),
    ((*T, "learning_rate"), 0, "[modes.fast.template].learning_rate"),
    ((*T, "count_limit"), 300, "[modes.fast.template].count_limit"),
    ((*T, "weight_max"), 2**50, "[modes.fast.template].weight_max"),
    ((*T, "hash_mul"), 0, "[modes.fast.template].hash_mul"),
    ((*T, "adaptive_start"), 10, "[modes.fast.template].adaptive_start"),
    ((*T, "word_context"), 1, "word_context"),
    ((*T, "prob_bits"), 17, "[modes.fast.template].prob_bits"),
    ((*T, "prob_bits"), 1, "[modes.fast.template].prob_bits"),
    ((*T, "fix_bits"), 25, "[modes.fast.template].fix_bits"),
    ((*T, "fix_bits"), 8, "[modes.fast.template].fix_bits"),
    ((*T, "adaptive_max"), 65536, "[modes.fast.template].adaptive_max"),
    ((*T, "squash_knots"), list(range(1, 35)), "[modes.fast.template].squash_knots"),
    ((*B, "kind"), "zipf", "[modes.fast.base].kind"),
    ((*B, "zeta"), 1.5, "[modes.fast.base].zeta"),
    ((*B, "u_scale_bits"), 60, "[modes.fast.base].u_scale_bits"),
    ((*B, "zipf_offset"), 0, "[modes.fast.base].zipf_offset"),
    ((*B, "table"), "ghost", "[modes.fast.base].table"),
    ((*M, "kind"), "fancy", "[modes.fast.mixer].kind"),
    ((*M, "eta"), 2.0, "[modes.fast.mixer].eta"),
    ((*M, "alpha"), -0.1, "[modes.fast.mixer].alpha"),
    ((*P, "corpus_pairs", "kind"), "trigram", "[modes.fast.predictors.corpus_pairs].kind"),
    ((*P, "corpus_pairs", "discount"), 1.5, "[modes.fast.predictors.corpus_pairs].discount"),
    ((*P, "corpus_pairs", "trust0"), 0.0, "[modes.fast.predictors.corpus_pairs].trust0"),
    ((*P, "text_pairs", "confidence_k"), 0.0, "[modes.fast.predictors.text_pairs].confidence_k"),
    ((*P, "text_pairs", "discount"), 0.5, "not used by kind = 'text_pairs'"),
    ((*G, "generic_codec"), "lz4", "[modes.generic].generic_codec"),
    ((*G, "word_model"), "null", "[modes.generic].word_model"),
    ((*G, "codecs"), ["lz4"], "[modes.generic].codecs[0]"),
    ((*G, "codecs"), [], "[modes.generic].codecs"),
    ((*G, "ppmd_memory_mb"), 0, "[modes.generic].ppmd_memory_mb"),
    ((*X, "word_coder"), "letters", "[modes.max].word_coder"),
    ((*X, "gap", "kind"), "commas", "[modes.max.gap].kind"),
    ((*X, "gap", "groups"), [0, 1, 999, 2, 2, 2, 2, 2], "[modes.max.gap].groups"),
    ((*XP, "text_gap_starts", "groups"), [1, 1], "[modes.max.predictors.text_gap_starts].groups"),
    ((*XP, "corpus_gap_starts", "discount"), 2.0, "[modes.max.predictors.corpus_gap_starts].discount"),
    ((*XP, "corpus_gap_starts", "window"), 3, "not used by kind = 'corpus_gap_starts'"),
    ((*XP, "prefix_class", "prior_table"), "ghost", "[modes.max.predictors.prefix_class].prior_table"),
    ((*XP, "prefix_class", "prior_add"), 0.0, "[modes.max.predictors.prefix_class].prior_add"),
    ((*XP, "recency", "window"), 0, "[modes.max.predictors.recency].window"),
    ((*X, "speller", "q_bits"), 0, "[modes.max.speller].q_bits"),
    ((*X, "speller", "trie"), "ghost", "[modes.max.speller].trie"),
    ((*X, "speller", "inputs"), ["cross_word", "within_word"], "[modes.max.speller].inputs"),
    ((*XC, "cross", "context"), "both", "[modes.max.letter_chains.cross].context"),
    ((*XC, "cross", "max_order"), 12, "[modes.max.letter_chains.cross].max_order"),
    ((*XC, "within", "table"), "ghost", "[modes.max.letter_chains.within].table"),
    ((*XC, "cross", "symbol_bits"), 9, "[modes.max.letter_chains.cross].symbol_bits"),
    ((*XL, "cross_word", "chain"), "ghost", "no such [letter_chains.*]"),
    ((*XL, "text_letters", "gamma"), 0.0, "[modes.max.letters.text_letters].gamma"),
    ((*XL, "text_letters", "chain"), "cross", "not used by kind = 'text_letters'"),
    ((*XL, "within_word", "kind"), "bigram", "[modes.max.letters.within_word].kind"),
    ((*X, "letter_mixer", "kind"), "linear", "[modes.max.letter_mixer].kind"),
    ((*X, "letter_mixer", "lr"), 2.0, "[modes.max.letter_mixer].lr"),
    ((*X, "letter_mixer", "start_weights"), [1.0], "[modes.max.letter_mixer].start_weights"),
    ((*X, "letter_mixer", "conf_buckets"), [0.9, 0.3], "[modes.max.letter_mixer].conf_buckets"),
    ((*X, "letter_mixer", "start_sets"), "ghost", "[modes.max.letter_mixer].start_sets"),
]
MISSING: list[tuple[Path, str]] = [
    ((*T, "conf_level"), "[modes.fast.template].conf_level"),
    ((*T, "prob_bits"), "[modes.fast.template].prob_bits"),
    ((*T, "fix_bits"), "[modes.fast.template].fix_bits"),
    ((*B, "zeta"), "[modes.fast.base].zeta"),
    ((*M, "eta"), "[modes.fast.mixer].eta"),
    ((*P, "corpus_pairs", "table"), "[modes.fast.predictors.corpus_pairs].table"),
    ((*F, "table_scale_bits"), "[modes.fast].table_scale_bits"),
    ((*G, "ppmd_memory_mb"), "[modes.generic].ppmd_memory_mb"),
    ((*F, "id"), "id"),
    ((*X, "speller"), "[modes.max.speller]"),
    ((*X, "letter_mixer", "lr"), "[modes.max.letter_mixer].lr"),
    ((*XP, "recency", "window"), "[modes.max.predictors.recency].window"),
    ((*XP, "prefix_class", "prior_k"), "[modes.max.predictors.prefix_class].prior_k"),
    ((*X, "gap", "groups"), "[modes.max.gap].groups"),
    ((*XC, "cross", "beta"), "beta"),
]


@pytest.mark.parametrize(("path", "value", "needle"), BAD, ids=[".".join(map(str, p)) + f"={v!r}"[:40]
                                                                 for p, v, _ in BAD])
def test_invalid_mode_row_is_refused_and_named(path: Path, value: Any, needle: str) -> None:
    cfg = real_config()
    set_path(cfg, path, value)
    assert needle in refused(cfg)


@pytest.mark.parametrize(("path", "needle"), MISSING, ids=[".".join(map(str, p)) for p, _ in MISSING])
def test_missing_drawer_row_is_refused(path: Path, needle: str) -> None:
    cfg = real_config()
    node: Any = cfg
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    assert needle in refused(cfg)


@pytest.mark.parametrize(("path", "plate"), [((*T, "kind"), "flat"), ((*B, "kind"), "uniform"),
                                             ((*M, "kind"), "static"), ((*P, "text_pairs", "kind"), "null"),
                                             ((*F, "word_model"), "null"), ((*F, "tokenizer"), "literal"),
                                             ((*X, "word_coder"), "direct"), ((*X, "gap", "kind"), "none"),
                                             ((*XP, "prefix_class", "kind"), "null"),
                                             ((*XL, "text_letters", "kind"), "null"),
                                             ((*X, "letter_mixer", "kind"), "word_only")])
def test_switching_to_the_plate_is_one_word(path: Path, plate: str) -> None:
    cfg = real_config()
    set_path(cfg, path, plate)
    load_dict(cfg)  # the real kind's numbers may stay: a plate reads none of them


def test_missing_drawer_word_is_allowed_it_means_the_plate() -> None:
    cfg = real_config()
    del cfg["modes"]["fast"]["tokenizer"]
    del cfg["modes"]["fast"]["template"]["kind"]
    settings = load_dict(cfg)
    assert settings.modes["fast"].tokenizer is None
