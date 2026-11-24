"""The alphabetical index of the dictionary = an implicit letter tree.

Every prefix is a contiguous range of ranks. ``perm[r]`` = the word id of rank r, ``rank_of`` the
inverse; ``cols[k][r]`` = the code of letter k of the word at rank r (0 = the word ended) for the
first ``cols.shape[0]`` letters - a table (``[tables.trie_he]``), checked against the dictionary.
Letter codes: 0 = word end, 1..L = the language's letters in code-point order (``LanguagePack.alphabet``).
Children of the node [lo, hi) at depth k = one range per next symbol. Small ranges (and depths past
the table's columns) are split by looking at the words themselves - the result is the same, only
faster (``SMALL_RANGE`` changes no bit).
"""

from __future__ import annotations

import operator
import re

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch

IArr = npt.NDArray[np.int64]
SMALL_RANGE = 24
"""Ranges up to this size are split in Python (speed only; both ways give the same children)."""
END = 0


class AlphaIndex:
    """Rank <-> word id, and the children of a tree node."""

    def __init__(self, perm: IArr, cols: npt.NDArray[np.uint8], words: list[str], alphabet: str) -> None:
        n = len(words)
        if len(perm) != n or cols.ndim != 2 or cols.shape[1] != n:
            raise TableMismatch(f"טבלת העץ אינה בגודל המילון ({n} מילים)")
        self.cmap = {ch: i + 1 for i, ch in enumerate(alphabet)}
        letters = re.compile(f"[{re.escape(alphabet)}]*" if alphabet else "")
        if letters.fullmatch("".join(words)) is None:  # one C-level scan (was a set of 8.5M letters)
            raise TableMismatch("במילון יש אות שאינה באלפבית של השפה")
        self.symbols = len(alphabet) + 1
        self.perm = perm
        self.cols = cols
        self.n = n
        self.rank_of = np.empty(n, dtype=np.int64)
        self.rank_of[perm] = np.arange(n, dtype=np.int64)
        self.by_id = words
        self.words = list(map(words.__getitem__, perm.tolist()))
        if not all(map(operator.lt, self.words, self.words[1:])):  # C-level pairwise compare
            raise TableMismatch("טבלת העץ אינה בסדר אלפביתי של המילון")
        self._steps = np.arange(self.symbols + 1, dtype=np.uint8)

    def codes(self, x: int) -> list[int]:
        """The letter codes of word id ``x`` (without the end)."""
        return [self.cmap[ch] for ch in self.by_id[x]]

    def word_at(self, rank: int) -> int:
        """The word id of a rank."""
        return int(self.perm[rank])

    def children(self, lo: int, hi: int, k: int) -> tuple[list[int], list[int]]:
        """(codes, bounds) of the non-empty children of [lo, hi) at depth k; ``len(bounds) = len(codes) + 1``."""
        if hi - lo <= SMALL_RANGE or k >= self.cols.shape[0]:
            return self._scan(lo, hi, k)
        b = (np.searchsorted(self.cols[k][lo:hi], self._steps) + lo).tolist()
        codes, bounds = [], []
        for c in range(self.symbols):
            if b[c + 1] > b[c]:
                codes.append(c)
                bounds.append(b[c])
        bounds.append(hi)
        return codes, bounds

    def _scan(self, lo: int, hi: int, k: int) -> tuple[list[int], list[int]]:
        cm = self.cmap
        codes: list[int] = []
        bounds: list[int] = []
        last = -1
        for r in range(lo, hi):
            w = self.words[r]
            c = cm[w[k]] if len(w) > k else END
            if c != last:
                codes.append(c)
                bounds.append(r)
                last = c
        bounds.append(hi)
        return codes, bounds
