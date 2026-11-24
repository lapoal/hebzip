"""Predictor drawer ``corpus_classes``: the next word given the CLASS of a word before it.

Why (word classes, a known PAQ-family idea; the best single gain measured: -0.9% on short texts): ``corpus_pairs``
knows only words it saw before in the corpus; after a rare word it has no follower list and the whole
mass goes to the base. Words with similar fastText vectors share a class (k-means, ``ops/classes.py``),
and the corpus counts "class of the word ``window`` back -> next word" give such a rare word the
followers of its whole class. Same shape as ``corpus_gap_starts``: one corpus list per context (here
the class), absolute discount D, the rest of the mass backs off to the base (escape), evidence 1 when
the class has a list. ``window`` = which earlier word gives the class (1 = the previous word, 2 = the
one before it: the research prototype's ``class_skip``); the history keeps two words, so 1 or 2 (start-up check).
Same expressions as the other corpus lists (``notebook``) - every integer of F(x) is exact.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import NO_EVIDENCE, Look, Sparse, WordHistory
from hebzip.domain.words.notebook import cumulative, discounted_escape, discounted_prob, discounted_sparse

IArr = npt.NDArray[np.int64]


class ClassLists:
    """Per class: the slice ``off[c]:off[c+1]`` of the sorted follower ids and their counts, and N[c]."""

    def __init__(self, off: IArr, ids: IArr, cnt: IArr, total: IArr) -> None:
        self.off, self.ids, self.cnt, self.total = off, ids, cnt, total
        self.kept = cumulative(cnt)

    def at(self, c: int) -> tuple[IArr, IArr, int, int, int] | None:
        """(ids, counts, N, kept count, list size) of class ``c``; ``None`` when the corpus never saw it."""
        n = int(self.total[c])
        if not n:
            return None
        lo, hi = int(self.off[c]), int(self.off[c + 1])
        return self.ids[lo:hi], self.cnt[lo:hi], n, int(self.kept[hi] - self.kept[lo]), hi - lo


class CorpusClasses:
    """P(x | class of the word ``window`` back) from the corpus, with back-off to the base."""

    name = "corpus_classes"
    active = True

    def __init__(self, classes: IArr, lists: ClassLists, window: int, discount: float, trust0: float) -> None:
        self.classes = classes
        self.lists = lists
        self.window = window
        self.discount = discount
        self.trust0 = trust0
        self.cur: tuple[IArr, IArr, int, int, int] | None = None

    def start(self) -> None:
        """Static knowledge - nothing to forget."""
        self.cur = None

    def look(self, h: WordHistory) -> Look:
        """Evidence 1 when the context word's class has a corpus list; escape = the discounted rest."""
        w = h.back(self.window)
        self.cur = self.lists.at(int(self.classes[w])) if w is not None else None
        if self.cur is None:
            return NO_EVIDENCE
        _, _, n, kept, size = self.cur
        return Look(1.0, discounted_escape(n, kept, size, self.discount), found=True)

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """max(floor((count - D) * w * scale / N), 0) per listed follower."""
        if self.cur is None:
            return None
        ids, cnt, n, _, _ = self.cur
        return discounted_sparse(ids, cnt, self.discount, weight, scale, n)

    def prob(self, x: int) -> float:
        """max(c - D, 0) / N for a listed follower, else 0."""
        if self.cur is None:
            return 0.0
        ids, cnt, n, _, _ = self.cur
        return discounted_prob(ids, cnt, x, self.discount, n)

    def observe(self, x: int, h: WordHistory) -> None:
        """The corpus does not learn."""
