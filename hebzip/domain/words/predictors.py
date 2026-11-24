"""Predictor drawers of mode fast (mixer channels) and the plate.

``corpus_pairs``  P(x | previous word) from the corpus, absolute discount D, the rest of the mass
                  backs off to the base (``escape`` = bo). No context for this word = escape 1.
``text_pairs``    the text's own (previous -> next) counts so far.
``text_counts``   the text's own word counts so far.
``null``          the plate: inactive - the mixture leaves it out of the mixer altogether.
Numeric rules: model maths in Python floats, written as the same expressions in the
same order as the research prototype; integer weights by truncation (int(c * f), floor((c - D) * f)).
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch
from hebzip.domain.contracts import NO_EVIDENCE, Look, Sparse, WordHistory
from hebzip.domain.words.notebook import (
    SortedCounts,
    TextChannel,
    count_one,
    cumulative,
    discounted_escape,
    discounted_prob,
    discounted_sparse,
    scaled_counts,
)

IArr = npt.NDArray[np.int64]
Ints = npt.NDArray[Any]
"""A table array of any integer width - the pinned one, a narrowed one, maybe a read-only map."""


class CorpusPairs:
    """Corpus word-pair prior with back-off to the base.

    The five table arrays come as they are stored (uint32 pinned, narrowed or mapped - no
    int64 copy at start-up), so no width may reach a number: every value leaves as a Python ``int`` or a
    float, and the follower ids are summed in int64 (``np.cumsum`` of uint32 is uint64, which mixed with
    int64 becomes float64 - a numba/numpy typing trap; guarded by ``tests/domain/test_narrow_tables.py``)."""

    name = "corpus_pairs"
    active = True

    def __init__(self, ctx: Ints, ctx_n: Ints, off: Ints, nxt_d: Ints, cnt: Ints, discount: float,
                 trust0: float) -> None:
        if len(ctx) > 1 and not bool(np.all(ctx[1:] > ctx[:-1])):
            raise TableMismatch("טבלת זוגות-המילים: ההקשרים אינם ממוינים")
        self.ctx_list: list[int] = ctx.tolist()
        self.ctx_n, self.off, self.nxt_d, self.cnt = ctx_n, off, nxt_d, cnt
        self.discount = discount
        self.trust0 = trust0
        self.at: tuple[int, int, int, IArr] | None = None

    def start(self) -> None:
        """Static knowledge - nothing to forget."""
        self.at = None

    def look(self, h: WordHistory) -> Look:
        """Find the previous word's follower list; escape = the discounted mass left for the base."""
        self.at = None
        if h.prev is not None:
            i = bisect_left(self.ctx_list, h.prev)
            if i < len(self.ctx_list) and self.ctx_list[i] == h.prev:
                lo, hi = int(self.off[i]), int(self.off[i + 1])
                big_n = int(self.ctx_n[i])
                # the research prototype's np.add.reduceat quirk: an empty slice "sums" to cnt[lo]
                cs = int(self.cnt[lo:hi].sum()) if hi > lo else int(self.cnt[lo])
                self.at = (lo, hi, big_n, np.cumsum(self.nxt_d[lo:hi], dtype=np.int64))
                return Look(1.0, discounted_escape(big_n, cs, hi - lo, self.discount), found=True)
        return Look(1.0, 1.0)

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """floor((count - D) * w * scale / N) per follower (never below 0)."""
        if self.at is None:
            return None
        lo, hi, big_n, ids = self.at
        return discounted_sparse(ids, self.cnt[lo:hi], self.discount, weight, scale, big_n)

    def prob(self, x: int) -> float:
        """max(c - D, 0) / N for a follower, else 0."""
        if self.at is None:
            return 0.0
        lo, hi, big_n, ids = self.at
        return discounted_prob(ids, self.cnt[lo:hi], x, self.discount, big_n)

    def observe(self, x: int, h: WordHistory) -> None:
        """The corpus does not learn."""


class TextPairs(TextChannel):
    """The text's own word pairs."""

    name = "text_pairs"
    active = True

    def start(self) -> None:
        """Empty notebook."""
        self.big: dict[int, dict[int, int]] = {}
        self.nb: dict[int, int] | None = None
        self.n_p = 0

    def look(self, h: WordHistory) -> Look:
        """Evidence = n_p / (n_p + k) where n_p = times the previous word was followed so far."""
        self.nb = self.big.get(h.prev) if h.prev is not None else None
        self.n_p = sum(self.nb.values()) if self.nb else 0
        return Look(self.n_p / (self.n_p + self.k) if self.n_p else 0.0, 0.0)

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(c * w * scale / n_p) per follower seen in this text."""
        if not self.n_p or self.nb is None:
            return None
        return scaled_counts(self.nb, weight, scale, self.n_p)

    def prob(self, x: int) -> float:
        """c / n_p."""
        return self.nb.get(x, 0) / self.n_p if self.n_p and self.nb is not None else 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """Count (previous, x)."""
        if h.prev is not None:
            count_one(self.big.setdefault(h.prev, {}), x)


class TextCounts(TextChannel):
    """The text's own word counts."""

    name = "text_counts"
    active = True

    def start(self) -> None:
        """Empty notebook."""
        self.counts = SortedCounts()
        self.uni: dict[int, int] = {}
        self.n = 0

    def look(self, h: WordHistory) -> Look:
        """Evidence = n / (n + k) where n = words so far."""
        return Look(self.n / (self.n + self.k) if self.n else 0.0, 0.0)

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(c * w * scale / n) per word seen in this text (float64 multiply, truncation)."""
        if not self.n:
            return None
        ids, cnt = self.counts.arrays()
        return Sparse(ids, cumulative((cnt * (weight * scale / self.n)).astype(np.int64)))

    def prob(self, x: int) -> float:
        """c / n."""
        return self.uni.get(x, 0) / self.n if self.n else 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """Count x."""
        count_one(self.uni, x)
        self.counts.add(x)
        self.n += 1


class NullPredictor:
    """The plate: no evidence, never part of the mixer (``active = False``)."""

    name = "null"
    active = False
    trust0 = 0.0

    def start(self) -> None:
        """Nothing."""

    def look(self, h: WordHistory) -> Look:
        """No evidence, no escape."""
        return NO_EVIDENCE

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """Nothing."""
        return None

    def prob(self, x: int) -> float:
        """Nothing."""
        return 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """Nothing."""
