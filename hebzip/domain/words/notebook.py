"""Small exact helpers for the word predictors: sorted counts, cumulative sums, the shared formulas.

Why sorted arrays kept up to date word by word: the coding table needs the text's word ids sorted
with their counts; re-sorting a dict for every word would cost O(m log m) each time.
Why the formulas live here (gate ``tests/gates/test_no_duplicate_math.py``): the text-table
step ``int(c * (w * scale / n))`` and the absolute-discount back-off of a corpus list were each
written twice (fast's and max's predictors). One copy each now, same expressions in the same order
as before, so every integer of F(x) is unchanged.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import Sparse

IArr = npt.NDArray[np.int64]
START = 256
"""First capacity of ``SortedCounts`` (it doubles when full - memory only, no bit changes)."""


def cumulative(weights: IArr) -> IArr:
    """[0, w0, w0+w1, ...] in exact int64."""
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(weights, dtype=np.int64)])


def count_one(table: dict[int, int], x: int) -> None:
    """One more ``x``."""
    table[x] = table.get(x, 0) + 1


def scaled_counts(counts: dict[int, int], weight: float, scale: int, n: int) -> Sparse:
    """``int(c * (weight * scale / n))`` per id, ids sorted (the text-table step)."""
    f = weight * scale / n
    ids = sorted(counts)
    return Sparse(np.array(ids, dtype=np.int64),
                  cumulative(np.array([int(counts[x] * f) for x in ids], dtype=np.int64)))


def scaled_arrays(ids: IArr, cnt: IArr, weight: float, scale: int, n: int) -> Sparse:
    """The same text-table step over arrays: ``int(c * (weight * scale / n))`` per id (``ids`` sorted).

    Why a second form: ``text_topic`` keeps up to window x neighbours ids, and building the
    Python list of ``scaled_counts`` for thousands of ids per word is slow. numpy multiplies the same two float64
    operands per id and ``astype(int64)`` truncates toward zero like ``int()`` (every product is >= 0), so the
    integers are the ones ``scaled_counts`` would give. ``scaled_counts`` itself is left as it was (its modes'
    bytes are pinned)."""
    f = weight * scale / n
    return Sparse(ids, cumulative((cnt.astype(np.float64) * f).astype(np.int64)))


def discounted_escape(total: int, kept: int, size: int, discount: float) -> float:
    """The mass a corpus list with absolute discount D leaves for the base: (N - (kept - D*size)) / N."""
    return (total - (kept - discount * size)) / total


def discounted_sparse(ids: IArr, cnt: IArr, discount: float, weight: float, scale: int, total: int) -> Sparse:
    """max(floor((count - D) * w * scale / N), 0) per listed id."""
    f = weight * scale / total
    cw = np.floor((cnt.astype(np.float64) - discount) * f).astype(np.int64)
    return Sparse(ids, cumulative(np.maximum(cw, 0)))


def discounted_prob(ids: IArr, cnt: IArr, x: int, discount: float, total: int) -> float:
    """max(c - D, 0) / N for a listed id, else 0."""
    j = int(np.searchsorted(ids, x))
    c = int(cnt[j]) if j < len(ids) and int(ids[j]) == x else 0
    return max(c - discount, 0.0) / total if c else 0.0


class TextChannel:
    """What every predictor that learns from the text alone is born with: k, its start trust, a reset."""

    def __init__(self, confidence_k: float, trust0: float) -> None:
        self.k = confidence_k
        self.trust0 = trust0
        self.start()

    def start(self) -> None:
        """Overridden: forget the previous message."""


class SortedCounts:
    """Counts per id, kept sorted by id."""

    def __init__(self) -> None:
        self.ids = np.zeros(START, dtype=np.int64)
        self.cnt = np.zeros(START, dtype=np.int64)
        self.m = 0

    def add(self, x: int) -> None:
        """Count one more ``x``."""
        m = self.m
        k = int(np.searchsorted(self.ids[:m], x))
        if k < m and int(self.ids[k]) == x:
            self.cnt[k] += 1
            return
        if m == len(self.ids):
            self.ids = np.concatenate([self.ids, np.zeros(m, dtype=np.int64)])
            self.cnt = np.concatenate([self.cnt, np.zeros(m, dtype=np.int64)])
        self.ids[k + 1:m + 1] = self.ids[k:m].copy()
        self.cnt[k + 1:m + 1] = self.cnt[k:m].copy()
        self.ids[k] = x
        self.cnt[k] = 1
        self.m = m + 1

    def arrays(self) -> tuple[IArr, IArr]:
        """(ids, counts) of every id seen, sorted by id."""
        return self.ids[:self.m], self.cnt[:self.m]
