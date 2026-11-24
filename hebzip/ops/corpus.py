"""A corpus as the table recipes see it: numbered documents, the held-out split, tokenized documents.

Why documents and line numbers: the research prototype's recipes read one article per line (``jsonl``,
field ``text``), held out every ``held_every``-th LINE (counting blank lines too: ``i % held_every ==
held_every - 1``) and never let a context reach across an article. The same rule here makes a rebuilt
Hebrew table comparable with the pinned one; a plain-text corpus is one document per line (``lines``).
The reader is chosen by a word (``READERS``), never by an if. Zero I/O: the caller hands in the text.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from hebzip.core.errors import InputError
from hebzip.domain.contracts import DictionaryPort, TokenizerPort
from hebzip.ops.counts import Doc

JSON_FIELD = "text"
"""The article field of a ``jsonl`` corpus (the research prototype's ``hewiki_random.jsonl``)."""
Numbered = list[tuple[int, str]]


def _lines(text: str) -> Numbered:
    return [(i, line) for i, line in enumerate(text.split("\n")) if line.strip()]


def _jsonl(text: str) -> Numbered:
    out: Numbered = []
    for i, line in _lines(text):
        try:
            out.append((i, str(json.loads(line)[JSON_FIELD])))
        except (ValueError, KeyError, TypeError) as error:
            raise InputError(f"שורה {i + 1} בקורפוס אינה JSON עם שדה '{JSON_FIELD}'") from error
    return out


READERS: dict[str, Callable[[str], Numbered]] = {"lines": _lines, "jsonl": _jsonl}
"""Corpus format word -> its reader (``--corpus-format``)."""


def is_held(number: int, held_every: int) -> bool:
    """Is document ``number`` held out (0 = nothing is)?"""
    return held_every > 0 and number % held_every == held_every - 1


@dataclass(frozen=True)
class Tokenized:
    """Every document of the corpus, tokenized with one tokenizer and one word list."""

    numbers: list[int]
    docs: list[Doc]

    def part(self, held_every: int, held: bool) -> list[Doc]:
        """The training documents (``held=False``) or the held-out ones, in corpus order."""
        return [d for n, d in zip(self.numbers, self.docs, strict=True) if is_held(n, held_every) == held]


def tokenize(numbered: Numbered, tokenizer: TokenizerPort, dictionary: DictionaryPort) -> Tokenized:
    """Split every document into (template, word ids)."""
    return Tokenized([n for n, _ in numbered], [tokenizer.split(text, dictionary) for _, text in numbered])
