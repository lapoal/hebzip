"""Word classes from word vectors, and the corpus class lists of ``corpus_classes``.

``kmeans``: cosine k-means on the ``fit_words`` first vectors (the word list is in frequency order, so
these are the most frequent words), ``iterations`` rounds from ``classes`` centroids drawn with ``seed``;
an empty cluster is re-seeded from a random word. Then EVERY word id gets its nearest centroid.
Why only the frequent words fit the centroids: rare-word vectors are noisy, and the
whole matrix (1.4 GB as float32) does not fit in memory; the rest are only assigned, in chunks.
Determinism: fixed seed, numpy only - the result is a pinned table, both sides read the file (a rebuild
on another machine may differ in a float of the BLAS sum; the sha256 row then says so).
``class_lists``: per window w (1 = the previous word, 2 = the one before it) and class c, the words that
followed a word of class c w words back in a training document; N{w}[c] = every such pair, the list keeps
pairs seen at least ``min_count`` times (pairs seen once stay in the back-off, like the pruned word pairs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.words.notebook import cumulative
from hebzip.ops.counts import Arrays, Doc

FArr = npt.NDArray[np.float32]
IArr = npt.NDArray[np.int64]
FIT_CHUNK = 50_000
ASSIGN_CHUNK = 100_000
"""Rows per matrix product (memory only - no value depends on them)."""
NORM_FLOOR = 1e-6
"""A zero vector stays zero instead of dividing by zero (a small floor)."""


def unit_rows(a: npt.NDArray[Any]) -> FArr:
    a32: FArr = a.astype(np.float32)
    n = np.linalg.norm(a32, axis=1, keepdims=True)
    out: FArr = (a32 / np.maximum(n, np.float32(NORM_FLOOR))).astype(np.float32, copy=False)
    return out


def _nearest(x: FArr, centroids: FArr, chunk: int) -> IArr:
    out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = np.argmax(x[s:s + chunk] @ centroids.T, axis=1)
    return out


def kmeans(vectors: npt.NDArray[Any], classes: int, fit_words: int, iterations: int, seed: int) -> IArr:
    """A class 0..classes-1 for every row of ``vectors`` (one row per word id)."""
    fit = min(fit_words, len(vectors))
    x = unit_rows(np.asarray(vectors[:fit]))
    rng = np.random.default_rng(seed)
    centroids = x[np.sort(rng.choice(fit, classes, replace=False))].copy()
    for _ in range(iterations):
        a = _nearest(x, centroids, FIT_CHUNK)
        total = np.zeros_like(centroids)
        np.add.at(total, a, x)
        empty = np.flatnonzero(np.bincount(a, minlength=classes) == 0)
        total[empty] = x[rng.choice(fit, len(empty), replace=False)]
        centroids = unit_rows(total)
    out = np.empty(len(vectors), dtype=np.int64)
    for s in range(0, len(vectors), ASSIGN_CHUNK):
        out[s:s + ASSIGN_CHUNK] = _nearest(unit_rows(np.asarray(vectors[s:s + ASSIGN_CHUNK])), centroids, ASSIGN_CHUNK)
    return out


def _pairs(docs: list[Doc], window: int) -> tuple[IArr, IArr]:
    before: list[int] = []
    after: list[int] = []
    for _template, ids in docs:
        before += ids[:-window]
        after += ids[window:]
    return np.array(before, dtype=np.int64), np.array(after, dtype=np.int64)


def _lists(cls: IArr, nxt: IArr, classes: int, size: int, min_count: int) -> tuple[IArr, IArr, IArr, IArr]:
    key = cls * size + nxt
    u, c = np.unique(key, return_counts=True)
    owner = u // size
    total = np.bincount(owner, weights=c, minlength=classes).astype(np.int64)
    keep = c >= min_count
    u, c, owner = u[keep], c[keep], owner[keep]
    off = cumulative(np.bincount(owner, minlength=classes).astype(np.int64))
    return off, (u % size).astype(np.int64), c.astype(np.int64), total


def class_lists(docs: list[Doc], word_class: IArr, windows: list[int], min_count: int) -> Arrays:
    """``off{w}``, ``ids{w}``, ``cnt{w}``, ``N{w}`` for every window ``w`` (sorted ids inside each class)."""
    classes = int(word_class.max()) + 1
    size = len(word_class)
    out: Arrays = {}
    for w in windows:
        before, after = _pairs(docs, w)
        off, ids, cnt, total = _lists(word_class[before], after, classes, size, min_count)
        out |= {f"off{w}": off, f"ids{w}": ids, f"cnt{w}": cnt, f"N{w}": total}
    return out
