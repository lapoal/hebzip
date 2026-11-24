"""Predictor drawers of mode max that learn only from the text itself.

``text_triples``     T3: (word before previous, previous) -> next, counted in this text.
``text_gap_starts``  TG: "first word after gap class g", counted in this text; it speaks only when
                     the gap's group is one of its ``groups`` rows (sentence start, after a mark).
``recency``          RC: the counts of the last ``window`` words.
Every one is the research prototype's component (``sp_model``) as its own drawer, with the same
expressions in the same order: evidence n / (n + k), probability c / n, integer
weights ``int(c * w * scale / n)`` per id (truncation) - so splitting the research prototype's one text
table into a sparse part per drawer changes no integer of F(x).
Their plate is ``NullPredictor`` (``predictors.py``): inactive, left out of the mixer.
"""

from __future__ import annotations

from collections import deque

from hebzip.domain.contracts import NO_EVIDENCE, Look, Sparse, WordHistory
from hebzip.domain.words.notebook import TextChannel
from hebzip.domain.words.notebook import count_one as _count
from hebzip.domain.words.notebook import scaled_counts as _sparse

MID_CLASS = 0


class TextTriples(TextChannel):
    """T3: the text's own word triples."""

    name = "text_triples"
    active = True

    def start(self) -> None:
        """Empty notebook."""
        self.tri: dict[tuple[int, int], dict[int, int]] = {}
        self.nb: dict[int, int] | None = None
        self.n3 = 0

    def look(self, h: WordHistory) -> Look:
        """Evidence n3 / (n3 + k), n3 = how often (prev2, prev) was followed so far."""
        self.nb, self.n3 = None, 0
        if h.prev2 is not None and h.prev is not None:
            self.nb = self.tri.get((h.prev2, h.prev))
            if self.nb:
                for c in self.nb.values():
                    self.n3 += c
                return Look(self.n3 / (self.n3 + self.k), 0.0)
        return NO_EVIDENCE

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(c * w * scale / n3) per follower."""
        return _sparse(self.nb, weight, scale, self.n3) if self.n3 and self.nb is not None else None

    def prob(self, x: int) -> float:
        """c / n3."""
        return self.nb.get(x, 0) / self.n3 if self.n3 and self.nb is not None else 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """Count (prev2, prev, x)."""
        if h.prev2 is not None and h.prev is not None:
            _count(self.tri.setdefault((h.prev2, h.prev), {}), x)


class TextGapStarts:
    """TG: the text's own first words after each gap class."""

    name = "text_gap_starts"
    active = True

    def __init__(self, confidence_k: float, trust0: float, groups: tuple[int, ...]) -> None:
        self.k = confidence_k
        self.trust0 = trust0
        self.groups = frozenset(groups)
        self.start()

    def start(self) -> None:
        """Empty notebook."""
        self.firsts: dict[int, dict[int, int]] = {}
        self.totals: dict[int, int] = {}
        self.at: dict[int, int] | None = None
        self.ng = 0

    def look(self, h: WordHistory) -> Look:
        """Evidence ng / (ng + k) for the gap's class, only when its group is switched on."""
        self.at, self.ng = None, 0
        if h.gap.group in self.groups:
            self.ng = self.totals.get(h.gap.cls, 0)
            if self.ng:
                self.at = self.firsts[h.gap.cls]
                return Look(self.ng / (self.ng + self.k), 0.0)
        return NO_EVIDENCE

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(c * w * scale / ng) per word seen after this gap class."""
        return _sparse(self.at, weight, scale, self.ng) if self.at is not None else None

    def prob(self, x: int) -> float:
        """c / ng."""
        return self.at.get(x, 0) / self.ng if self.at is not None else 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """Count x after its gap class (every class but mid - whatever ``groups`` says)."""
        if h.gap.cls != MID_CLASS:
            _count(self.firsts.setdefault(h.gap.cls, {}), x)
            self.totals[h.gap.cls] = self.totals.get(h.gap.cls, 0) + 1


class Recency:
    """RC: counts of the last ``window`` words."""

    name = "recency"
    active = True

    def __init__(self, window: int, confidence_k: float, trust0: float) -> None:
        self.window = window
        self.k = confidence_k
        self.trust0 = trust0
        self.start()

    def start(self) -> None:
        """Empty window."""
        self.queue: deque[int] = deque()
        self.counts: dict[int, int] = {}
        self.m = 0

    def look(self, h: WordHistory) -> Look:
        """Evidence m / (m + k), m = min(words so far, window)."""
        self.m = min(h.n, self.window)
        return Look(self.m / (self.m + self.k), 0.0) if self.m else NO_EVIDENCE

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(c * w * scale / m) per word in the window."""
        return _sparse(self.counts, weight, scale, self.m) if self.m else None

    def prob(self, x: int) -> float:
        """c / m."""
        return self.counts.get(x, 0) / self.m if self.m else 0.0

    def observe(self, x: int, h: WordHistory) -> None:
        """x enters the window; the oldest word leaves when it is full (a count of 0 is dropped)."""
        self.queue.append(x)
        _count(self.counts, x)
        if len(self.queue) > self.window:
            y = self.queue.popleft()
            self.counts[y] -= 1
            if not self.counts[y]:
                del self.counts[y]
