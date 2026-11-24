"""Which drawers a mode really uses, and which tables they read - derived from the rows, never listed.

Why: ``config/tables.py`` used to keep a hand-written list of table rows with two
if-branches on drawer words (``dictionary == "word_list"``, ``word_coder == "spelled"``); a new drawer
that reads a table meant editing that list. Now a mode is walked generically: every drawer family
(``FAMILIES``) gives a ``Use`` (family, word, rows); a drawer whose parent is switched off (its
``PARENT`` word is not chosen) is not used. The tables a used drawer reads are its rows named in
``TABLE_ROWS`` (``KIND_ROWS`` says which rows a kind has), plus the few reads that go through another
row (``LINKED_READS``: the dictionary reads the language's word list, a letter input reads its chain's
table). Adding a drawer that reads a table = rows in ``KIND_ROWS``; no code here changes.
Plates read nothing (their ``KIND_ROWS`` entry is empty). Zero I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from hebzip.config.checks import sub as _sub
from hebzip.config.schema import KIND_ROWS, PLATE, TABLE_ROWS, ModeRow, Settings

TOP: Final = ("dictionary", "tokenizer", "word_model", "word_coder", "generic_codec")
"""Families chosen by a word directly on ``[modes.<m>]`` (their rows are the mode row's own)."""
ONE: Final = {"template": "template", "base": "base", "mixer": "mixer", "gap": "gap", "letter_mixer": "letter_mixer"}
"""Family -> the mode row's field holding its one sub-table."""
MANY: Final = {"predictor": "predictors", "letter": "letters"}
"""Family -> the mode row's field holding its named sub-tables."""
PARENT: Final = {"base": ("word_model", "mixture"), "mixer": ("word_model", "mixture"),
                 "predictor": ("word_model", "mixture"), "gap": ("word_model", "mixture"),
                 "word_coder": ("word_model", "mixture"), "letter": ("word_coder", "spelled"),
                 "letter_mixer": ("word_coder", "spelled")}
"""Family -> (parent family, the parent word that uses it). A plate parent reads no child's rows."""


@dataclass(frozen=True)
class Use:
    """One drawer of one mode: its family, its word, where its rows live, its row object, the mode."""

    family: str
    kind: str
    where: str
    row: Any
    mode: ModeRow
    mode_name: str


def all_uses(name: str, row: ModeRow) -> list[Use]:
    """Every drawer row of the mode (used or not), words defaulted to the plate."""
    at = f"[modes.{name}]"
    out = [Use(f, getattr(row, f) or PLATE[f], at, row, row, name) for f in TOP]
    for family, field in ONE.items():
        sub = getattr(row, field)
        out.append(Use(family, (sub.kind if sub else None) or PLATE[family], _sub(at, field), sub, row, name))
    for family, field in MANY.items():
        for key, sub in getattr(row, field).items():
            out.append(Use(family, sub.kind or PLATE[family], _sub(at, f"{field}.{key}"), sub, row, name))
    return out


def uses(name: str, row: ModeRow) -> list[Use]:
    """The drawers the mode really uses (a child of a switched-off parent is left out)."""
    found = all_uses(name, row)
    words = {u.family: u.kind for u in found if u.family not in MANY}
    return [u for u in found if u.family not in PARENT or words[PARENT[u.family][0]] == PARENT[u.family][1]]


def _language_words(use: Use, settings: Settings) -> list[str]:
    return [settings.languages[use.mode.language].word_list] if use.mode.language in settings.languages else []


def _chain_table(link: str) -> Callable[[Use, Settings], list[str]]:
    def read(use: Use, settings: Settings) -> list[str]:
        chain = use.mode.letter_chains.get(getattr(use.row, link) or "")
        return [chain.table] if chain is not None else []
    return read


def _speller_tree(use: Use, settings: Settings) -> list[str]:
    return [use.mode.speller.trie] if use.mode.speller is not None else []


LINKED_READS: Final[dict[tuple[str, str], Callable[[Use, Settings], list[str]]]] = {
    ("dictionary", "word_list"): _language_words,
    ("word_coder", "spelled"): _speller_tree,
    ("letter", "static_chain"): _chain_table("chain"),
    ("letter", "text_letters"): _chain_table("base_chain"),
}
"""Table reads that go through another row (the language's word list, a named chain, the tree)."""


def reads(use: Use, settings: Settings) -> list[str]:
    """The ``[tables.*]`` names one drawer reads."""
    rows = KIND_ROWS.get(use.family, {}).get(use.kind, ())
    own = [getattr(use.row, r) for r in rows if r in TABLE_ROWS]
    linked = LINKED_READS.get((use.family, use.kind))
    return [n for n in own if n] + (linked(use, settings) if linked else [])


def table_readers(settings: Settings, modes: list[str] | None = None) -> dict[str, list[Use]]:
    """Table name -> every drawer (of the given modes, default all) that reads it, in config order."""
    out: dict[str, list[Use]] = {}
    for name in modes if modes is not None else list(settings.modes):
        for use in uses(name, settings.modes[name]):
            for table in reads(use, settings):
                out.setdefault(table, []).append(use)
    return out
