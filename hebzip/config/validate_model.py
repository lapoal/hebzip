"""Validation of one ``[modes.<m>]`` and its drawer sub-tables.

Why tables of "kind -> required rows" (``schema.KIND_ROWS``) and "row -> allowed range": a drawer word
decides which numbers it needs; a missing number is a start-up error naming the row, an extra one
(meant for another kind) too. Ranges keep every kernel inside its exact-integer limits:
e.g. ``weight_max`` <= 2**46 keeps the template mixer's dot product below 2**63 in both engines.
Why registries and no ``if kind == ...``: the checks that belong to one word
(the mode kinds, ``cm``, ``mixture``, ``best_of``, the codec families) are entries of a table keyed by
that word, so a new drawer = new rows + a new entry, never an edited if-chain. The walk over a mode's
drawers (which are used, which rows they have) is ``config/drawers.py``.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

from hebzip.config import schema as S
from hebzip.config.checks import bad, in_range, one_of
from hebzip.config.drawers import TOP, Use, all_uses, uses
from hebzip.config.validate_words import KIND_EXTRA_WORDS, ROW_CHECKS, WORD_EXTRA

U32 = 2**32 - 1
FIX_MAX = 24
"""The widest fixed point: with ``fix_bits`` <= 24 the adaptive step ``(2**fix_bits) * recip`` stays below
2**50 and the mixer step ``xs * err`` below 2**52 - far inside int64 in both engines."""
RANGES: dict[str, tuple[float, float]] = {
    "prob_bits": (2, 16), "fix_bits": (2, FIX_MAX),
    "word_distance_cap": (0, 255), "word_seed_order": (0, 65535), "static_scale": (1, 256),
    "conf_level": (0, 255), "learning_rate": (1, 4096), "count_limit": (1, 255), "adaptive_start": (0, 2**FIX_MAX - 1),
    "adaptive_min": (0, 2**FIX_MAX - 1), "adaptive_max": (0, 2**FIX_MAX - 1), "adaptive_recip_offset2": (1, 65536),
    "bias_input": (-4096, 4096), "weight_max": (1, 2**46), "hash_mul": (1, U32), "c0_mul": (1, U32),
    "seed_mul": (1, U32), "seed_add": (0, U32), "finish_mul": (1, U32), "finish_shift1": (1, 31),
    "finish_shift2": (1, 31),
    "zeta": (0.0, 1.0), "u_scale_bits": (1, 52), "zipf_weight_bits": (1, 52), "zipf_offset": (1, 2**31),
    "eta": (0.0, 1.0), "alpha": (0.0, 1.0), "discount": (0.0, 1.0), "confidence_k": (1e-12, 1e12),
    "trust0": (1e-12, 1e12), "table_scale_bits": (1, 61), "ppmd_memory_mb": (1, 2048),
    "window": (1, 2**20), "prior_k": (0.0, 1e12), "prior_add": (1e-12, 1e6), "gamma": (1e-12, 1e12),
    "max_order": (1, 11), "lr": (0.0, 1.0), "depth_max": (0, 255), "position_split": (1, 2**20),
}
MAX_ORDER = 32
ADAPTIVE_ROWS = ("adaptive_min", "adaptive_start", "adaptive_max")
ModeCheck = Callable[[str, str, S.ModeRow, S.Settings, str], None]


def check_mode(name: str, row: S.ModeRow, settings: S.Settings, source: str) -> None:
    """Everything about one mode."""
    at = f"[modes.{name}]"
    one_of(source, f"{at}.kind", row.kind, S.KINDS)
    in_range(source, f"{at}.id", row.id, 1, 2**28)
    for drawer, words in S.MODE_WORDS.items():
        value = getattr(row, drawer)
        if value is not None:
            one_of(source, f"{at}.{drawer}", value, words)
    MODE_KIND_CHECKS[row.kind](name, at, row, settings, source)


def _check_model(name: str, at: str, row: S.ModeRow, settings: S.Settings, source: str) -> None:
    _refuse_present(at, row, ("generic_codec", "ppmd_memory_mb", "candidates"), source)
    if row.codecs:
        raise bad(source, f"{at}.codecs", row.codecs, "only with kind = 'generic'")
    if row.language not in settings.languages:
        raise bad(source, f"{at}.language", row.language, "no such [languages.*] (a model mode names its language)")
    for use in uses(name, row):
        if use.family in TOP:
            extra = WORD_EXTRA.get((use.family, use.kind))
            if extra is not None:
                extra(use, settings, source, check_drawer)
        else:
            check_drawer(use, settings, source)


def _refuse_present(at: str, row: object, names: tuple[str, ...], source: str) -> None:
    for name in names:
        if getattr(row, name) is not None:
            raise bad(source, f"{at}.{name}", getattr(row, name), "not allowed here")


def check_drawer(use: Use, settings: S.Settings, source: str) -> None:
    """Kind word known; exactly its rows present; numbers in range; table names exist; the kind's own checks."""
    at, family, kind = use.where, use.family, use.kind
    one_of(source, f"{at}.kind", kind, tuple(S.KIND_ROWS[family]))
    if kind == S.PLATE[family]:
        return  # a plate reads no row: switching a drawer off is ONE word, its numbers may stay
    needed = S.KIND_ROWS[family][kind]
    for f in dataclasses.fields(use.row):
        value = getattr(use.row, f.name)
        if f.name == "kind":
            continue
        if f.name in needed and value is None:
            raise bad(source, f"{at}.{f.name}", None, f"missing (needed by kind = '{kind}')")
        if f.name not in needed and value is not None:
            raise bad(source, f"{at}.{f.name}", value, f"not used by kind = '{kind}'")
        if value is not None and f.name in RANGES:
            in_range(source, f"{at}.{f.name}", value, *RANGES[f.name])
        if value is not None and f.name in S.TABLE_ROWS and value not in settings.tables:
            raise bad(source, f"{at}.{f.name}", value, "no such [tables.*]")
        row_check = ROW_CHECKS.get((family, f.name))
        if value is not None and row_check is not None:
            row_check(use, f"{at}.{f.name}", value, source)
    extra = KIND_EXTRA.get((family, kind))
    if extra is not None:
        extra(use, source)


def _check_cm(use: Use, source: str) -> None:
    at, t = use.where, use.row
    orders, adaptive = t.orders or [], t.adaptive or []
    if not orders or len(set(orders)) != len(orders) or any(not 0 <= o <= MAX_ORDER for o in orders):
        raise bad(source, f"{at}.orders", orders, f"distinct orders 0..{MAX_ORDER}, at least one")
    if len(set(adaptive)) != len(adaptive) or any(not 0 <= o <= MAX_ORDER for o in adaptive):
        raise bad(source, f"{at}.adaptive", adaptive, f"distinct orders 0..{MAX_ORDER}")
    _check_precision(use, source)
    if not (t.adaptive_min or 0) <= (t.adaptive_start or 0) <= (t.adaptive_max or 0):
        raise bad(source, f"{at}.adaptive_start", t.adaptive_start, "needs adaptive_min <= start <= adaptive_max")


def _check_precision(use: Use, source: str) -> None:
    """The rows that live in the precision's units: adaptive probabilities are read at ``prob_bits``
    (so ``fix_bits`` >= ``prob_bits``) and are ``fix_bits`` wide; the knots are probabilities in ``prob_bits``,
    2**m + 1 of them (the knot spacing ``2**(prob_bits - m)`` must be a whole number of bits)."""
    at, t = use.where, use.row
    prob_bits, fix_bits, knots = int(t.prob_bits or 0), int(t.fix_bits or 0), t.squash_knots or []
    if fix_bits < prob_bits:
        raise bad(source, f"{at}.fix_bits", fix_bits, f"at least prob_bits ({prob_bits})")
    one, spans = 1 << prob_bits, len(knots) - 1
    shape_ok = spans >= 2 and not spans & (spans - 1) and spans <= one
    if not shape_ok or any(not 1 <= k < one for k in knots) or knots != sorted(knots):
        raise bad(source, f"{at}.squash_knots", knots,
                  f"2**m + 1 rising probabilities 1..{one - 1} (m = 1..prob_bits)")
    for name in ADAPTIVE_ROWS:
        if getattr(t, name) >= 1 << fix_bits:
            raise bad(source, f"{at}.{name}", getattr(t, name), f"below 2**fix_bits ({1 << fix_bits})")


def _check_generic(name: str, at: str, row: S.ModeRow, settings: S.Settings, source: str) -> None:
    for field in MODEL_ONLY + ("candidates",):
        _refuse_present(at, row, (field,), source)
    if row.channels or row.predictors:
        raise bad(source, f"{at}.channels", row.channels, "not allowed with kind = 'generic'")
    for use in all_uses(name, row):
        extra = WORD_EXTRA.get((use.family, use.kind))
        if use.family in TOP and extra is not None:
            extra(use, settings, source, check_drawer)
    families: set[str] = set()
    for i, codec in enumerate(row.codecs):
        family = codec.split(":")[0]
        form = S.CODEC_FORMS.get(family)
        if form is None or not re.fullmatch(form, codec):
            raise bad(source, f"{at}.codecs[{i}]", codec, "not a codec word (raw, deflate:9, xz:9:extreme, ...)")
        families.add(family)
    for family in sorted(families):
        for needed in S.CODEC_ROWS.get(family, ()):
            if getattr(row, needed) is None:
                raise bad(source, f"{at}.{needed}", None, f"missing (needed by a {family} codec)")
    if row.ppmd_memory_mb is not None:
        in_range(source, f"{at}.ppmd_memory_mb", row.ppmd_memory_mb, *RANGES["ppmd_memory_mb"])


def _check_choose(name: str, at: str, row: S.ModeRow, settings: S.Settings, source: str) -> None:
    """A choose mode: only ``id`` + ``candidates``; every candidate a live model/generic mode, each once."""
    for field in MODEL_ONLY + ("word_coder", "gap", "speller", "letter_mixer", "generic_codec", "ppmd_memory_mb"):
        _refuse_present(at, row, (field,), source)
    if row.channels or row.predictors or row.codecs or row.letters or row.letter_chains:
        raise bad(source, f"{at}.channels", row.channels, "a choose mode has only id + candidates")
    cands = row.candidates or []
    if not cands or len(set(cands)) != len(cands):
        raise bad(source, f"{at}.candidates", row.candidates, "one or more distinct mode names")
    for c in cands:
        other = settings.modes.get(c)
        if other is None or other.retired or not S.CHOOSE_KINDS.get(other.kind, False):
            raise bad(source, f"{at}.candidates", c, "must name a live mode with kind = 'model' or 'generic'")


MODEL_ONLY = ("language", "dictionary", "tokenizer", "word_model", "table_scale_bits", "template", "base", "mixer")
"""Rows only a model mode may have."""
MODE_KIND_CHECKS: dict[str, ModeCheck] = {"model": _check_model, "generic": _check_generic, "choose": _check_choose}
"""Mode kind -> its checks (every word of ``schema.KINDS`` is a key - tested)."""
KIND_EXTRA: dict[tuple[str, str], Callable[[Use, str], Any]] = {("template", "cm"): _check_cm, **KIND_EXTRA_WORDS}
"""(family, kind) -> checks beyond "its rows, in range" (the letter mixer's live in ``validate_words``)."""
