"""Config rows of the max_plus drawers and of the ``choose`` mode kind: every wrong row is refused
at start-up naming the row; every row is in the signature.

The signature rule: every row that affects bytes is signed, present or absent - the rows added
for ``max_plus`` are in the signature of every mode, and changing one changes it.
``choose`` (compress with every candidate, keep the smaller; the header names the candidate): no mode of the real
config uses it today, so its rows are checked on a copy of the config with one test-only mode ``pick``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from hebzip.config.load import load_dict
from hebzip.services.select import _rows
from tests.config.test_load import Path, refused, set_path
from tests.support import SIGNATURE_BYTE, real_config

PLUS = "max_plus"
P, PP, PM = ("modes", PLUS), ("modes", PLUS, "predictors"), ("modes", PLUS, "mixer")
C = ("modes", "pick")
PICK = {"id": 9, "kind": "choose", "candidates": ["fast", "max"]}
"""A test-only choose mode (``tests/services/test_plus.py`` runs the same one through the CLI)."""
BAD: list[tuple[Path, Any, str]] = [
    ((*PP, "class_pairs", "window"), 3, "[modes.max_plus.predictors.class_pairs].window"),
    ((*PP, "class_pairs", "window"), 0, "[modes.max_plus.predictors.class_pairs].window"),
    ((*PP, "class_pairs", "prior_table"), "ghost", "[modes.max_plus.predictors.class_pairs].prior_table"),
    ((*PP, "class_pairs", "discount"), 1.5, "[modes.max_plus.predictors.class_pairs].discount"),
    ((*PM, "select_channel"), "text_pairs", "[modes.max_plus.mixer].select_channel"),
    ((*PM, "select_channel"), "ghost", "[modes.max_plus.mixer].select_channel"),
    ((*PM, "position_split"), 0, "[modes.max_plus.mixer].position_split"),
    ((*PM, "trust_sets"), "ghost", "[modes.max_plus.mixer].trust_sets"),
    ((*C, "candidates"), [], "[modes.pick].candidates"),
    ((*C, "candidates"), ["fast", "fast"], "[modes.pick].candidates"),
    ((*C, "candidates"), ["ghost"], "[modes.pick].candidates"),
    ((*C, "candidates"), ["pick"], "[modes.pick].candidates"),
    ((*C, "language"), "he", "[modes.pick].language"),
    ((*P, "candidates"), ["max"], "[modes.max_plus].candidates"),
    (("modes", "generic", "candidates"), ["max"], "[modes.generic].candidates"),
    (("tables", "word_classes", "recipe", "classes"), 1, "[tables.word_classes.recipe].classes"),
    (("hebzip", "fallback"), ["pick"], "[hebzip].fallback"),
    ((*PP, "topic", "window"), 0, "[modes.max_plus.predictors.topic].window"),
    ((*PP, "topic", "table"), "ghost", "[modes.max_plus.predictors.topic].table"),
    ((*PP, "topic", "discount"), 0.5, "[modes.max_plus.predictors.topic].discount"),
    (("tables", "topic_neighbours", "recipe", "skip_top"), 300000, "[tables.topic_neighbours.recipe].skip_top"),
    (("tables", "topic_neighbours", "recipe", "neighbours"), 0, "[tables.topic_neighbours.recipe].neighbours"),
    (("tables", "topic_neighbours", "recipe", "sim_floor"), 1.0, "[tables.topic_neighbours.recipe].sim_floor"),
    (("tables", "topic_neighbours", "recipe", "held_every"), 40, "[tables.topic_neighbours.recipe].held_every"),
    (("tables", "topic_trust_sets", "recipe", "append"), [], "[tables.topic_trust_sets.recipe].append"),
    (("tables", "topic_trust_sets", "recipe", "append"), [0.0], "[tables.topic_trust_sets.recipe].append"),
    (("tables", "topic_trust_sets", "recipe", "source"), "ghost", "[tables.topic_trust_sets.recipe].source"),
]


def with_pick() -> dict[str, Any]:
    """The real config plus the test-only choose mode ``pick``."""
    cfg = real_config()
    cfg["modes"]["pick"] = dict(PICK)
    return cfg


def test_the_choose_mode_loads() -> None:
    settings = load_dict(with_pick())
    assert settings.modes["pick"].candidates == ["fast", "max"] and settings.modes["pick"].language is None


@pytest.mark.parametrize(("path", "value", "needle"), BAD, ids=[".".join(map(str, p)) for p, _, _ in BAD])
def test_invalid_plus_row_is_refused_and_named(path: Path, value: Any, needle: str) -> None:
    cfg = with_pick()
    set_path(cfg, path, value)
    assert needle in refused(cfg)


def test_a_missing_plus_row_is_refused() -> None:
    cfg = real_config()
    del cfg["modes"][PLUS]["mixer"]["position_split"]
    assert "[modes.max_plus.mixer].position_split" in refused(cfg)
    cfg = with_pick()
    del cfg["modes"]["pick"]["candidates"]
    assert "[modes.pick].candidates" in refused(cfg)
    cfg = real_config()
    del cfg["modes"][PLUS]["predictors"]["topic"]["confidence_k"]
    assert "[modes.max_plus.predictors.topic].confidence_k" in refused(cfg)


def _names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {n for v in value.values() for n in _names(v)}
    return set()


LATER_ROWS = {"trust_sets", "select_channel", "position_split", "candidates"}
"""Rows added after the first modes' signatures were fixed. Earlier they were left out of the
signature when absent; now every row is signed, present or absent."""


def test_every_row_is_signed_present_or_absent() -> None:
    settings = load_dict(real_config())
    for mode in SIGNATURE_BYTE:
        every = _names(dataclasses.asdict(settings.modes[mode])) - {"retired", "language"}
        assert _names(_rows(settings, mode)["mode"]) == every, f"{mode}: a row is missing from the signature"
    fast = _rows(settings, "fast")["mode"]
    assert isinstance(fast, dict) and "candidates" in fast and fast["candidates"] is None, "an absent row is signed"
    assert LATER_ROWS <= _names(_rows(settings, PLUS)), "a present row is signed"
    cfg = real_config()
    cfg["modes"][PLUS]["mixer"]["position_split"] = 11
    assert _rows(load_dict(cfg), PLUS) != _rows(settings, PLUS)
