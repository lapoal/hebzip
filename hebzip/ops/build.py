"""One table build: the corpus of one language, the tables its modes read, and what is already made.

Why lazy: a recipe may need another table - the start weights need the static prior, the class
prior needs the classes, every word table needs the word list. ``value`` makes a table from its
recipe the first time it is asked (and keeps it), or READS the pinned file when the table has no
recipe (it was given: the Hebrew word list, for one). Which recipe makes a table is its row
``[tables.<name>.recipe].kind`` -> the ``recipes`` table handed in (``ops/recipes.py``); which drawer
reads it, and so which mode rows it must follow, comes from ``config/drawers.py``. The drawers a
recipe needs (tokenizer, gap classifier) come from the one selector (``ModeBuild``), python engine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.config.drawers import Use, table_readers
from hebzip.config.schema import PLATE, LanguageRow, RecipeRow, Settings
from hebzip.core.errors import ConfigError
from hebzip.domain.contracts import KernelSet
from hebzip.domain.dictionary import WordList
from hebzip.ops.corpus import Numbered, Tokenized, tokenize
from hebzip.ops.counts import Arrays
from hebzip.services import assemble as A
from hebzip.services.assemble import SETS_KEY
from hebzip.services.select import ModeBuild, engine_kernels, table_store
from hebzip.store.files import read_mapped

BUILD_ENGINE = "python"
"""The builder applies the hash kernels to whole arrays - only the plain-Python kernels can."""
BUILD_STORE = "eager"
"""The builder reads the PINNED tables: a derived copy may be stale while tables are rebuilt."""


@dataclass(frozen=True)
class Made:
    """A table made in memory: its value (what readers get) and how to write it to a file."""

    value: Any
    write: Callable[[Path], None]


Recipe = Callable[["Build", str, RecipeRow], Made]


class Build:
    """The tables of one language's modes, made from one corpus."""

    def __init__(self, settings: Settings, language: str, numbered: Numbered, root: Path,
                 recipes: Mapping[str, Recipe], vectors: Path | None = None) -> None:
        if language not in settings.languages:
            raise ConfigError(f"[languages.{language}]: no such language")
        self.settings = dataclasses.replace(settings, hebzip=dataclasses.replace(
            settings.hebzip, engine=BUILD_ENGINE, table_store=BUILD_STORE))
        self.language = language
        modes = [m for m, row in settings.modes.items() if row.language == language]
        self.readers = table_readers(self.settings, modes)
        self.numbered = numbered
        self.root = root
        self.recipes = recipes
        self.vectors_path = vectors
        self.store = table_store(self.settings, root)
        self.loader = A.Loader(self.store)
        self.made: dict[str, Made] = {}
        self._modes: dict[str, ModeBuild] = {}
        self._tokenized: dict[str, Tokenized] = {}

    def lang(self) -> LanguageRow:
        """The language's rows."""
        return self.settings.languages[self.language]

    def recipe(self, name: str) -> RecipeRow | None:
        """The table's recipe row (``None`` = a given table)."""
        table = self.settings.tables.get(name)
        if table is None:
            raise ConfigError(f"[tables.{name}]: no such table")
        return table.recipe

    def make(self, name: str) -> Made:
        """The table made from its recipe (once)."""
        if name not in self.made:
            recipe = self.recipe(name)
            if recipe is None:
                raise ConfigError(f"[tables.{name}]: no [tables.{name}.recipe] - this table is given, not built")
            self.made[name] = self.recipes[recipe.kind](self, name, recipe)
        return self.made[name]

    def _value(self, name: str, read: Callable[[str], Any]) -> Any:
        return self.make(name).value if self.recipe(name) is not None else read(name)

    def words(self, name: str) -> list[str]:
        """A word-list table: made, or read when given."""
        words: list[str] = self._value(name, self.loader.words)
        return words

    def arrays(self, name: str) -> Arrays:
        """An ``.npz`` table: made, or read when given."""
        arrays: Arrays = dict(self._value(name, self.loader.arrays))
        return arrays

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """An ``.npy`` table: made, or read when given."""
        array: npt.NDArray[np.generic] = self._value(name, self.loader.array)
        return array

    def weight_sets(self, name: str) -> dict[tuple[int, int], list[float]]:
        """A letter mixer's start-set table (``{SETS_KEY: {"d,c": [...]}}``): made, or read when given."""
        if self.recipe(name) is None:
            return self.loader.weight_sets(name, SETS_KEY)
        made: dict[str, dict[str, list[float]]] = self.make(name).value
        return {(int(k.split(",")[0]), int(k.split(",")[1])): list(v) for k, v in made[SETS_KEY].items()}

    def vectors(self, name: str) -> npt.NDArray[np.generic]:
        """The word vectors (``--vectors``: one row per word id, memory-mapped) - a build input like the corpus."""
        if self.vectors_path is None:
            raise ConfigError(f"[tables.{name}]: its recipe needs the word vectors - give --vectors PATH (.npy)")
        return read_mapped(self.vectors_path)

    def reader(self, name: str, agreed: bool = False) -> Use:
        """The (first) drawer that reads the table. ``agreed``: the table's content follows the reader's rows
        (template orders, gap groups), so every drawer that reads it must have the same rows."""
        found = self.readers.get(name)
        if not found:
            raise ConfigError(f"[tables.{name}]: no mode of [languages.{self.language}] reads it")
        first = found[0]
        for other in found[1:] if agreed else []:
            if other.row != first.row:
                raise ConfigError(f"[tables.{name}]: {first.where} and {other.where} read it with different rows "
                                  "- give each its own table")
        return first

    def mode(self, use: Use) -> ModeBuild:
        """The selector's builder of the reading drawer's mode (tokenizer, gap classifier ...)."""
        if use.mode_name not in self._modes:
            self._modes[use.mode_name] = ModeBuild(self.settings, use.mode_name, self.root, self.store)
        return self._modes[use.mode_name]

    def kernels(self) -> KernelSet:
        """The python-engine kernels."""
        return engine_kernels(BUILD_ENGINE, self.root)

    def tokenized(self, use: Use) -> Tokenized:
        """The corpus split by the reading mode's tokenizer and the language's word list (kept per tokenizer)."""
        tokenizer = self.mode(use).tokenizer()
        key = use.mode.tokenizer or PLATE["tokenizer"]
        if key not in self._tokenized:
            words = self.words(self.lang().word_list)
            self._tokenized[key] = tokenize(self.numbered, tokenizer, WordList(words))
        return self._tokenized[key]
