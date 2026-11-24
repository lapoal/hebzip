"""Predictor drawer ``text_topic``: the next word by its TOPIC - fastText neighbours of the text's recent content words.

Why (measured: ranking candidate words by fastText similarity to the content words around them made the
rare "far" words of one text ~27% cheaper): no corpus channel knows that a text about lighthouses will say
"lens" and "coast" - the word pairs know only the word right before. Here every content word of the text so far
(a word that HAS a neighbour list: the table leaves the most frequent words and the very rare ones without one)
brings its nearest words in the vector space, and the channel predicts in proportion to the summed weights:

    acc[x] = sum over the last ``window`` content words y of  weight(y -> x)      P(x) = acc[x] / sum(acc)

Exactness (both sides compute the same numbers): the table (``ops/topics.py``) holds the neighbour
lists as INTEGERS (similarity quantised once, at build time, into 1..levels); at run time ``acc`` is integer
addition and subtraction only, and the coding weights are the text-table step ``int(c * (w * scale / n))`` that
every text channel uses (``notebook.scaled_arrays``) - one float multiply per id, same operands on both sides,
and the decoder sees exactly the words the encoder saw before each word (``observe`` comes after coding).
No float similarity is computed while coding.
Evidence like the other text channels: m / (m + k), m = content words in the window; escape 0 (the mixer's
weight decides how much of the table this channel gets). Its plate is ``NullPredictor`` (left out of the mixer).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import NO_EVIDENCE, Look, Sparse, WordHistory
from hebzip.domain.words.notebook import scaled_arrays

IArr = npt.NDArray[np.int64]
Ints = npt.NDArray[np.integer]


class NeighbourLists:
    """Per word id: the slice ``off[x]:off[x+1]`` of its neighbour ids (sorted) and their integer weights."""

    def __init__(self, off: Ints, ids: Ints, wt: Ints) -> None:
        self.off, self.ids, self.wt = off, ids, wt

    def at(self, x: int) -> tuple[list[int], list[int]]:
        """(neighbour ids, weights) of word ``x`` as Python ints - empty when the word has no list."""
        lo, hi = int(self.off[x]), int(self.off[x + 1])
        return [int(v) for v in self.ids[lo:hi]], [int(v) for v in self.wt[lo:hi]]


class TextTopic:
    """P(x | the topic of the last ``window`` content words) from integer neighbour lists."""

    name = "text_topic"
    active = True

    def __init__(self, lists: NeighbourLists, window: int, confidence_k: float, trust0: float) -> None:
        self.lists = lists
        self.window = window
        self.k = confidence_k
        self.trust0 = trust0
        self.start()

    def start(self) -> None:
        """No topic yet."""
        self.queue: deque[int] = deque()
        self.acc: dict[int, int] = {}
        self.total = 0
        self.sorted: tuple[IArr, IArr] | None = None

    def look(self, h: WordHistory) -> Look:
        """Evidence m / (m + k), m = content words in the window (none -> no evidence)."""
        if not self.total:
            return NO_EVIDENCE
        m = len(self.queue)
        return Look(m / (m + self.k), 0.0)

    def _arrays(self) -> tuple[IArr, IArr]:
        if self.sorted is None:
            ids = sorted(self.acc)
            self.sorted = (np.array(ids, dtype=np.int64), np.array([self.acc[x] for x in ids], dtype=np.int64))
        return self.sorted

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """int(acc[x] * (w * scale / total)) per id the topic knows."""
        if not self.total:
            return None
        ids, cnt = self._arrays()
        return scaled_arrays(ids, cnt, weight, scale, self.total)

    def prob(self, x: int) -> float:
        """acc[x] / total."""
        return self.acc.get(x, 0) / self.total if self.total else 0.0

    def _move(self, y: int, sign: int) -> None:
        ids, wts = self.lists.at(y)
        for x, w in zip(ids, wts, strict=True):
            c = self.acc.get(x, 0) + sign * w
            if c:
                self.acc[x] = c
            else:
                del self.acc[x]
            self.total += sign * w

    def observe(self, x: int, h: WordHistory) -> None:
        """A content word enters the window (its list is added); the oldest leaves when it is full."""
        if int(self.lists.off[x + 1]) == int(self.lists.off[x]):
            return
        self.queue.append(x)
        self._move(x, 1)
        if len(self.queue) > self.window:
            self._move(self.queue.popleft(), -1)
        self.sorted = None
