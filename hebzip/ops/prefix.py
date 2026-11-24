"""The proclitic-class recipes: ``prefix_classes`` (a class per word) and ``prefix_prior`` (class transitions).

Why here and from rows: the drawer ``prefix_class`` only READS a class per word
id; how a word gets its class was the research prototype's ``sp_model.prefix_class`` with the Hebrew letters
written in its code. Now the letters are the language row ``prefixes`` (which no code read before) and
the shape numbers are recipe rows: words up to ``short_len`` letters are class 1; a word starting with
the i-th proclitic letter is class 2+i; a word of at least ``double_min_len`` letters whose first letter
is the d-th of ``double_first`` and whose second is the i-th proclitic is class 2+P*(1+d)+i; anything
else is class 0. With Hebrew's rows that is exactly the research prototype's 16 classes. Zero I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import GapClassifierPort, TokenizerPort
from hebzip.ops.counts import Doc, slot_gaps

NONE_CLASS = 0
SHORT_CLASS = 1
FIRST_CLASS = 2
"""Class numbers of the recipe (the class table's layout, read back only as numbers)."""


def class_count(prefixes: str, double_first: str) -> int:
    """How many classes the rows give: none, short, one per proclitic, one per (double letter, proclitic)."""
    return FIRST_CLASS + len(prefixes) * (1 + len(double_first))


def word_class(word: str, prefixes: str, double_first: str, short_len: int, double_min_len: int) -> int:
    """The proclitic class of one word (see the module text)."""
    if len(word) <= short_len:
        return SHORT_CLASS
    first = prefixes.find(word[0])
    if first < 0:
        return NONE_CLASS
    double = double_first.find(word[0])
    if double >= 0 and len(word) >= double_min_len:
        second = prefixes.find(word[1])
        if second >= 0:
            return FIRST_CLASS + len(prefixes) * (1 + double) + second
    return FIRST_CLASS + first


def class_array(words: list[str], prefixes: str, double_first: str, short_len: int,
                double_min_len: int) -> npt.NDArray[np.generic]:
    """One class per word id, in the smallest unsigned type that holds every class."""
    count = class_count(prefixes, double_first)
    dtype = np.uint8 if count <= 1 << 8 else np.uint16
    return np.array([word_class(w, prefixes, double_first, short_len, double_min_len) for w in words], dtype=dtype)


def prior_counts(docs: list[Doc], tokenizer: TokenizerPort, gap: GapClassifierPort,
                 classes: npt.NDArray[np.generic], count: int) -> npt.NDArray[np.int64]:
    """Counts [previous class (``count`` = none yet)][gap group][class] over the documents' slots."""
    out = np.zeros((count + 1, gap.groups, count), dtype=np.int64)
    for template, ids in docs:
        prev = count
        for found, x in zip(slot_gaps(template, tokenizer, gap), ids, strict=True):
            cls = int(classes[x])
            out[prev, found.group, cls] += 1
            prev = cls
    return out
