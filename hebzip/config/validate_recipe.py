"""Validation of ``[tables.<name>.recipe]`` - how ``ops/build_langpack`` makes a table.

Why rows: every number of a table recipe (the held-out split, the minimum count of the template
prior, its confidence levels, how much text the start weights are fitted on, the shape of the
proclitic classes ...) is something another language may want different - a row, checked at start-up
like every other row (the change-without-code rule). Same rule as the drawers: the recipe word (``kind``) decides
which rows it needs (``schema.RECIPE_ROWS``); a missing one or one meant for another kind is refused.
"""

from __future__ import annotations

import dataclasses

from hebzip.config import schema as S
from hebzip.config.checks import bad, in_range, one_of

LEVELS_MAX = 255
"""Confidence levels are stored as one byte per table entry."""
RANGES: dict[str, tuple[float, float]] = {
    "held_every": (0, 2**20), "keep_top": (1, 2**31), "min_count": (1, 2**31), "fit_cut": (1, 2**31),
    "fit_passes": (1, 1000), "start_weight": (-(2**40), 2**40), "short_len": (0, 64), "double_min_len": (2, 64),
    "columns": (1, 255), "classes": (2, 65535), "fit_words": (2, 2**31), "iterations": (1, 1000),
    "seed": (0, 2**32 - 1),
    "neighbours": (1, 4096), "skip_top": (0, 2**31), "candidates": (2, 2**31), "sim_floor": (-1.0, 0.999),
    "levels": (1, 65535),
}


def check_recipe(at: str, recipe: S.RecipeRow, source: str, tables: frozenset[str]) -> None:
    """Kind known; exactly its rows; numbers in range; levels rising; letters given once; a named table exists."""
    one_of(source, f"{at}.kind", recipe.kind, tuple(S.RECIPE_ROWS))
    needed = S.RECIPE_ROWS[recipe.kind]
    for f in dataclasses.fields(recipe):
        value = getattr(recipe, f.name)
        if f.name == "kind":
            continue
        if f.name in needed and value is None:
            raise bad(source, f"{at}.{f.name}", None, f"missing (needed by kind = '{recipe.kind}')")
        if f.name not in needed and value is not None:
            raise bad(source, f"{at}.{f.name}", value, f"not used by kind = '{recipe.kind}'")
        if value is not None and f.name in RANGES:
            in_range(source, f"{at}.{f.name}", value, *RANGES[f.name])
    levels = recipe.count_levels
    if levels is not None and (not levels or len(levels) > LEVELS_MAX or levels != sorted(set(levels))
                               or levels[0] < 1):
        raise bad(source, f"{at}.count_levels", levels, f"1..{LEVELS_MAX} rising counts, each >= 1")
    if recipe.source is not None and recipe.source not in tables:
        raise bad(source, f"{at}.source", recipe.source, "no such [tables.*]")
    halve = recipe.halve
    if halve is not None and (not halve or len(set(halve)) != len(halve) or min(halve) < 0):
        raise bad(source, f"{at}.halve", halve, "distinct input positions >= 0, at least one")
    if recipe.skip_top is not None and recipe.candidates is not None and recipe.skip_top >= recipe.candidates:
        raise bad(source, f"{at}.skip_top", recipe.skip_top, f"below candidates ({recipe.candidates})")
    append = recipe.append
    if append is not None and (not append or any(not 0.0 < v < 1e12 for v in append)):
        raise bad(source, f"{at}.append", append, "one or more start trusts > 0")
    first = recipe.double_first
    if first is not None and len(set(first)) != len(first):
        raise bad(source, f"{at}.double_first", first, "each letter once")
