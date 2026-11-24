"""Builders of the speller (word coder ``spelled``): tree, letter chains, letter mixer.

Why apart from ``select.py``: the selector decides WHICH letter drawer a word means; these functions
only know HOW to build one from its rows and tables, and check each table against what reads it:
the tree must fit the dictionary, a chain table must have every order its rows ask for (sorted
keys, one count per key, symbols fitting ``symbol_bits``), the start sets one weight per input.
The speller's letter history is as long as the longest context any of its inputs reads.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from hebzip.config.schema import ChainRow, LetterMixerRow, SpellerRow
from hebzip.core.errors import TableMismatch
from hebzip.domain.contracts import LetterInputPort, LetterMixerPort
from hebzip.domain.lang import LanguagePack
from hebzip.domain.spell.chains import FAMILY, WITHIN, LetterChain
from hebzip.domain.spell.inputs import StaticChainInput, TextLetters
from hebzip.domain.spell.mixer import GeometricMixer
from hebzip.domain.spell.speller import SpelledWordCoder
from hebzip.domain.spell.trie import AlphaIndex
from hebzip.services.assemble import SETS_KEY, Loader

TRIE_ARRAYS = ("perm", "cols")
COUNT_ARRAY = "n0"
CHAIN_ARRAYS = ("{}{}_ck", "{}{}_cn")
INT63 = 1 << 63
U8Arr = npt.NDArray[np.uint8]


def letter_chain(row: ChainRow, loader: Loader, lang: LanguagePack) -> LetterChain:
    """One static chain from its table: orders 1..max_order of its context family."""
    symbols = len(lang.alphabet()) + 1
    n0 = loader.int64(row.table, COUNT_ARRAY)
    if len(n0) != symbols or symbols > 1 << row.symbol_bits:
        raise TableMismatch(f"טבלת {row.table}: {len(n0)} סמלים, השפה צריכה {symbols}")
    if (symbols + 1) ** row.max_order << row.symbol_bits >= INT63:
        raise TableMismatch(f"טבלת {row.table}: הקשר בסדר {row.max_order} לא נכנס ל-63 ביט")
    entries = []
    family = FAMILY[row.context]
    for k in range(1, row.max_order + 1):
        ck, cn = (loader.ints(row.table, name.format(family, k)) for name in CHAIN_ARRAYS)
        if len(ck) != len(cn) or (len(ck) > 1 and not bool(np.all(ck[1:] > ck[:-1]))):
            raise TableMismatch(f"טבלת {row.table}: סדר {k} של {row.context} אינו ממוין")
        entries.append((ck, cn))
    return LetterChain(n0, entries, WITHIN[row.context], row.beta, row.add_count, row.symbol_bits)


def static_chain(chain: LetterChain) -> StaticChainInput:
    """The letter input ``static_chain``."""
    return StaticChainInput(chain)


def text_letters(base: LetterChain, max_order: int | None, gamma: float | None) -> TextLetters:
    """The letter input ``text_letters``."""
    assert max_order is not None and gamma is not None
    return TextLetters(base, max_order, gamma)


def geometric_mixer(row: LetterMixerRow, loader: Loader, mixed: list[bool]) -> GeometricMixer:
    """The letter mixer ``geometric`` with its pretrained start sets (a table)."""
    assert row.lr is not None and row.start_weights is not None and row.depth_max is not None
    assert row.conf_buckets is not None and row.start_sets is not None
    sets = loader.weight_sets(row.start_sets, SETS_KEY)
    if any(len(w) != len(mixed) for w in sets.values()):
        raise TableMismatch(f"טבלת {row.start_sets}: צריך {len(mixed)} משקלים בכל סט")
    return GeometricMixer(row.lr, list(row.start_weights), row.depth_max, list(row.conf_buckets), sets, mixed)


def history_length(inputs: list[LetterInputPort]) -> int:
    """The longest context any input reads (at least 1)."""
    longest = 1
    for inp in inputs:
        longest = max(longest, inp.reach)
    return longest


def speller(row: SpellerRow, loader: Loader, lang: LanguagePack, words: list[str], inputs: list[LetterInputPort],
            mixer: LetterMixerPort) -> SpelledWordCoder:
    """The word coder ``spelled``."""
    perm = loader.int64(row.trie, TRIE_ARRAYS[0])
    cols = loader.arrays(row.trie).get(TRIE_ARRAYS[1])
    if cols is None or cols.dtype != np.uint8:
        raise TableMismatch(f"טבלת {row.trie}: חסר מערך האותיות (uint8)")
    index = AlphaIndex(perm, cast(U8Arr, cols), words, lang.alphabet())  # no copy - uint8 checked, maybe a map
    return SpelledWordCoder(index, inputs, mixer, row.q_bits, history_length(inputs))
