"""Predictor drawers of mode max that read a corpus table.

``corpus_gap_starts``  CG: "first word after gap class g" from the corpus (the top words per class,
                       ``ids{g}`` / ``cnt{g}`` / ``N{g}``), absolute discount D, the rest of the mass
                       backs off to the base (``escape`` = bog) - the gap twin of ``corpus_pairs``.
                       It speaks only when the gap's group is one of its ``groups`` rows.
``prefix_class``       PX: Hebrew proclitic classes (the table ``classes`` gives each word id its
                       class). P(class | previous class, gap group) from the corpus prior (+ the
                       ``prior_add`` row, normalised per row) mixed with this text's own counts
                       (weight ``prior_k``), times P(word | class) from the text. Its sparse part
                       is ``int(count * fac[class])`` over every word seen so far.
Same expressions in the same order as the research prototype (``sp_model``) -
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import NO_EVIDENCE, Look, Sparse, WordHistory
from hebzip.domain.words.notebook import (
    SortedCounts,
    count_one,
    cumulative,
    discounted_escape,
    discounted_prob,
    discounted_sparse,
)

IArr = npt.NDArray[np.int64]
FArr = npt.NDArray[np.float64]


class GapList:
    """One gap class's corpus list: sorted ids, their counts, the class total N, kept count and size."""

    def __init__(self, ids: IArr, cnt: IArr, total: int) -> None:
        self.ids, self.cnt, self.total = ids, cnt, total
        self.kept = int(cnt.sum())
        self.size = len(ids)


class CorpusGapStarts:
    """CG: corpus first-words per gap class, with back-off to the base."""

    name = "corpus_gap_starts"
    active = True

    def __init__(self, lists: list[GapList], discount: float, trust0: float, groups: tuple[int, ...]) -> None:
        self.lists = lists
        self.discount = discount
        self.trust0 = trust0
        self.groups = frozenset(groups)
        self.at: GapList | None = None

    def start(self) -> None:
        """Static knowledge - nothing to forget."""
        self.at = None

    def look(self, h: WordHistory) -> Look:
        """Evidence 1 when the gap's group is on and its class has a corpus list; escape = bog."""
        self.at = None
        if h.gap.group in self.groups and self.lists[h.gap.cls].total:
            g = self.at = self.lists[h.gap.cls]
            return Look(1.0, discounted_escape(g.total, g.kept, g.size, self.discount), found=True)
        return NO_EVIDENCE

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """max(floor((count - D) * w * scale / N), 0) per listed word."""
        if self.at is None:
            return None
        g = self.at
        return discounted_sparse(g.ids, g.cnt, self.discount, weight, scale, g.total)

    def prob(self, x: int) -> float:
        """max(c - D, 0) / N for a listed word, else 0."""
        if self.at is None:
            return 0.0
        g = self.at
        return discounted_prob(g.ids, g.cnt, x, self.discount, g.total)

    def observe(self, x: int, h: WordHistory) -> None:
        """The corpus does not learn."""


class PrefixClass:
    """PX: proclitic class transitions x the text's words of that class."""

    name = "prefix_class"
    active = True

    def __init__(self, classes: IArr, prior: list[list[list[float]]], confidence_k: float, prior_k: float,
                 trust0: float) -> None:
        self.classes = classes
        self.prior = prior
        self.n_classes = len(prior[0][0])
        self.k = confidence_k
        self.prior_k = prior_k
        self.trust0 = trust0
        self.start()

    def start(self) -> None:
        """Empty notebook; the previous class is "none" (the prior's last row)."""
        self.counts = SortedCounts()
        self.uni: dict[int, int] = {}
        self.cls_tot = [0] * self.n_classes
        self.trans: dict[tuple[int, int], list[int]] = {}
        self.prev_cls = self.n_classes
        self.n = 0
        self.dist: list[float] | None = None

    def look(self, h: WordHistory) -> Look:
        """Evidence n / (n + k); the class distribution of the next word is kept for sparse/prob."""
        self.dist = None
        if not self.n:
            return NO_EVIDENCE
        self.dist = self._dist(h.gap.group)
        return Look(self.n / (self.n + self.k), 0.0)

    def _dist(self, group: int) -> list[float]:
        """P(class) over classes that have seen words: (text count + prior_k * prior) / z."""
        prior = self.prior[self.prev_cls][group]
        row = self.trans.get((self.prev_cls, group))
        v = [0.0] * self.n_classes
        z = 0.0
        for c in range(self.n_classes):
            if self.cls_tot[c]:
                v[c] = (row[c] if row else 0) + self.prior_k * prior[c]
                z += v[c]
        return [x / z for x in v]

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(count * (w * scale * P(class) / class total)) per word seen."""
        if self.dist is None:
            return None
        fac = np.array([weight * scale * self.dist[k] / self.cls_tot[k] if self.cls_tot[k] else 0.0
                        for k in range(self.n_classes)], dtype=np.float64)
        ids, cnt = self.counts.arrays()
        return Sparse(ids, cumulative((cnt * fac[self.classes[ids]]).astype(np.int64)))

    def prob(self, x: int) -> float:
        """P(class of x) * count(x) / class total."""
        u = self.uni.get(x, 0)
        if self.dist is None or not u:
            return 0.0
        k = int(self.classes[x])
        return self.dist[k] * u / self.cls_tot[k]

    def observe(self, x: int, h: WordHistory) -> None:
        """Count x, its class, and the class transition under the gap's group."""
        k = int(self.classes[x])
        self.cls_tot[k] += 1
        self.trans.setdefault((self.prev_cls, h.gap.group), [0] * self.n_classes)[k] += 1
        self.prev_cls = k
        count_one(self.uni, x)
        self.counts.add(x)
        self.n += 1


def normalised_prior(counts: npt.NDArray[np.generic], add: float) -> list[list[list[float]]]:
    """(counts + add), each [prev][group] row divided by its sum - the research prototype's ``px_prior``.

    The rows are integers + ``add``; with add = 0.5 every sum is exact in any order."""
    px: FArr = counts.astype(np.float64) + add
    return [[(row / row.sum()).tolist() for row in grp] for grp in px]
