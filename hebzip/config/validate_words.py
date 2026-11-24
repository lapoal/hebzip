"""Checks that belong to one drawer word or one row of a mixture mode.

Why apart from ``validate_model``: these are the per-word entries of its registries - the mode-level
words (``mixture``, ``spelled``, ``best_of``: ``WORD_EXTRA``), single rows that need more than a range
(``groups``, a letter input's chain name: ``ROW_CHECKS``) and the letter mixer (``KIND_EXTRA_WORDS``).
Each is keyed by the word or row it belongs to, so nothing here asks "which word is this".
Dependencies (the declared-dependency rule) are refused at start-up, naming the row: ``mixture`` needs the dictionary
``word_list`` (``NEEDS``); ``spelled`` needs ``[speller]``, ``[letter_mixer]`` and one ``[letters.<name>]``
per input; every chain a letter input names must exist; the mixer has one start weight per input
plus one for the word model itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hebzip.config import schema as S
from hebzip.config.checks import bad, in_range, one_of
from hebzip.config.checks import sub as _sub
from hebzip.config.drawers import Use

GROUP_MAX = 255
CHAIN_RANGES: dict[str, tuple[float, float]] = {"max_order": (1, 11), "beta": (1e-12, 1e12),
                                                "add_count": (1e-12, 1e6), "symbol_bits": (1, 8)}
Q_BITS = (1, 52)
SCALE_BITS = (1, 61)
DrawerCheck = Callable[[Use, S.Settings, str], None]
NEEDS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {("word_model", "mixture"): (("dictionary", "word_list"),)}
"""(family, word) -> the (family, word) pairs it needs on the same mode (the declared-dependency rule)."""


def _distinct_groups(use: Use, where: str, groups: Any, source: str) -> None:
    if len(set(groups)) != len(groups) or any(not 0 <= g <= GROUP_MAX for g in groups):
        raise bad(source, where, groups, f"distinct gap groups 0..{GROUP_MAX}")


def _class_groups(use: Use, where: str, groups: Any, source: str) -> None:
    """``punctuation``: every group 0..255. How MANY groups the language's gap classes need is the
    classifier's own layout - checked at start-up by the selector (``services/select.py``,
    ``check_fits``), so the class layout is written once (``test_no_duplicate_math``)."""
    if any(not 0 <= g <= GROUP_MAX for g in groups):
        raise bad(source, where, groups, f"every group must be 0..{GROUP_MAX}")


def _chain_named(use: Use, where: str, name: Any, source: str) -> None:
    if name not in use.mode.letter_chains:
        raise bad(source, where, name, "no such [letter_chains.*]")


ROW_CHECKS: dict[tuple[str, str], Callable[[Use, str, Any, str], None]] = {
    ("predictor", "groups"): _distinct_groups, ("gap", "groups"): _class_groups,
    ("letter", "chain"): _chain_named, ("letter", "base_chain"): _chain_named}
"""(family, row) -> the check of that row beyond its range."""


def _needs(use: Use, source: str) -> None:
    for family, word in NEEDS.get((use.family, use.kind), ()):
        have = getattr(use.mode, family) or S.PLATE[family]
        if have != word:
            raise bad(source, f"{use.where}.{family}", getattr(use.mode, family),
                      f"{use.family} = '{use.kind}' needs {family} = '{word}'")


def _mixture(use: Use, settings: S.Settings, source: str, drawer: DrawerCheck) -> None:
    row, at = use.mode, use.where
    _needs(use, source)
    if row.table_scale_bits is None:
        raise bad(source, f"{at}.table_scale_bits", None, f"missing (needed by word_model = '{use.kind}')")
    in_range(source, f"{at}.table_scale_bits", row.table_scale_bits, *SCALE_BITS)
    if len(set(row.channels)) != len(row.channels) or set(row.channels) != set(row.predictors):
        raise bad(source, f"{at}.channels", row.channels, "must list every [predictors.*] exactly once")


def _spelled(use: Use, settings: S.Settings, source: str, drawer: DrawerCheck) -> None:
    row, at = use.mode, use.where
    if row.speller is None:
        raise bad(source, f"{_sub(at, 'speller')}", None, f"missing (needed by word_coder = '{use.kind}')")
    _check_speller(at, row.speller, settings, source)
    if set(row.letters) != set(row.speller.inputs):
        raise bad(source, f"{_sub(at, 'speller')}.inputs", row.speller.inputs,
                  "must list every [letters.*] exactly once")
    for name, chain in row.letter_chains.items():
        _check_chain(_sub(at, f"letter_chains.{name}"), chain, settings, source)


def _best_of(use: Use, settings: S.Settings, source: str, drawer: DrawerCheck) -> None:
    if not use.mode.codecs:
        raise bad(source, f"{use.where}.codecs", use.mode.codecs, f"{use.kind} needs at least one codec")


WORD_EXTRA: dict[tuple[str, str], Callable[[Use, S.Settings, str, DrawerCheck], None]] = {
    ("word_model", "mixture"): _mixture, ("word_coder", "spelled"): _spelled, ("generic_codec", "best_of"): _best_of}
"""(mode-level family, word) -> what that word needs on the mode."""


def _check_speller(at: str, sp: S.SpellerRow, settings: S.Settings, source: str) -> None:
    where = _sub(at, "speller")
    if sp.trie not in settings.tables:
        raise bad(source, f"{where}.trie", sp.trie, "no such [tables.*]")
    in_range(source, f"{where}.q_bits", sp.q_bits, *Q_BITS)
    if len(set(sp.inputs)) != len(sp.inputs):
        raise bad(source, f"{where}.inputs", sp.inputs, "duplicate names")


def _check_chain(where: str, chain: S.ChainRow, settings: S.Settings, source: str) -> None:
    one_of(source, f"{where}.context", chain.context, S.CHAIN_CONTEXTS)
    if chain.table not in settings.tables:
        raise bad(source, f"{where}.table", chain.table, "no such [tables.*]")
    for name, (low, high) in CHAIN_RANGES.items():
        in_range(source, f"{where}.{name}", getattr(chain, name), low, high)


def _letter_mixer(use: Use, source: str) -> None:
    m, where = use.row, use.where
    inputs = len(use.mode.speller.inputs) if use.mode.speller is not None else 0
    weights = m.start_weights or []
    if len(weights) != inputs + 1:
        raise bad(source, f"{where}.start_weights", weights, f"one per input plus the word model ({inputs + 1})")
    buckets = m.conf_buckets or []
    if any(not 0.0 < b <= 1.0 for b in buckets) or buckets != sorted(set(buckets)):
        raise bad(source, f"{where}.conf_buckets", buckets, "rising thresholds in (0, 1]")


CLASS_WINDOWS = (1, 2)
"""``corpus_classes.window``: the word history keeps the previous word and the one before it."""


def _class_window(use: Use, source: str) -> None:
    if use.row.window not in CLASS_WINDOWS:
        raise bad(source, f"{use.where}.window", use.row.window, "1 (the previous word) or 2 (the one before it)")


FOUND_KINDS: dict[str, bool] = {"corpus_pairs": True, "corpus_gap_starts": True, "corpus_classes": True}
"""Predictor kinds that say whether they found a corpus list (``Look.found``) - what ``trust_selected`` reads."""


def _selected_channel(use: Use, source: str) -> None:
    name = use.row.select_channel
    prow = use.mode.predictors.get(name) if name in use.mode.channels else None
    if prow is None or not FOUND_KINDS.get(prow.kind or "", False):
        raise bad(source, f"{use.where}.select_channel", name,
                  "must name a channel of this mode whose kind is " + " / ".join(FOUND_KINDS))


KIND_EXTRA_WORDS: dict[tuple[str, str], Callable[[Use, str], None]] = {("letter_mixer", "geometric"): _letter_mixer,
                                                                      ("predictor", "corpus_classes"): _class_window,
                                                                      ("mixer", "trust_selected"): _selected_channel}
"""(family, kind) -> checks of a sub-drawer beyond its rows."""
