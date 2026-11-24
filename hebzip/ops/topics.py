"""Neighbour lists from word vectors for the predictor ``text_topic``.

For every word id ``y`` in ``skip_top .. candidates - 1`` (the word list is in frequency order): its ``neighbours``
most similar words x in the same id range, x != y, by cosine of the vectors, kept when cosine >= ``sim_floor``,
each with the INTEGER weight

    round(levels * (cos - sim_floor) / (1 - sim_floor)), at least 1

Every other word id gets an empty list (it neither brings a topic nor is predicted by one).
Why integers made here and not cosines at run time: the coder must compute the same numbers on both sides;
float dot products through BLAS are not promised to repeat bit for bit (thread split, alignment), a pinned
integer table is. The table is built once and pinned by its sha256 (a rebuild on another machine may differ in a
float of the BLAS sum near a tie; the sha256 row then says so - same rule as ``ops/classes.py``).
Determinism here: one fixed order of chunks; ``argpartition`` then a full sort by (-cosine, id) inside the kept
candidates, so equal cosines are ordered by id; the output slice of a word is sorted by id (what the drawer reads).
Memory (the machine has 1-4 GB free): the vectors stay memory-mapped; only the candidate block as float32
(candidates x dim) and one chunk of similarities (``CHUNK`` x candidates) are in memory at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.words.notebook import cumulative
from hebzip.ops.classes import unit_rows
from hebzip.ops.counts import Arrays

FArr = npt.NDArray[np.float32]
IArr = npt.NDArray[np.int64]
CHUNK = 256
"""Rows of similarities computed at once (memory only - no value depends on it)."""


def _top(sims: FArr, k: int, floor: float) -> tuple[list[IArr], list[FArr]]:
    """Per row: the ``k`` best columns with cosine >= floor, best first, ties by column."""
    k = min(k, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows_ids: list[IArr] = []
    rows_cos: list[FArr] = []
    for r in range(len(sims)):
        cols = part[r]
        cos = sims[r, cols]
        order = np.lexsort((cols, -cos))
        cols, cos = cols[order], cos[order]
        keep = cos >= np.float32(floor)
        rows_ids.append(cols[keep].astype(np.int64))
        rows_cos.append(cos[keep])
    return rows_ids, rows_cos


def _weights(cos: FArr, floor: float, levels: int) -> IArr:
    w = np.rint(levels * (cos.astype(np.float64) - floor) / (1.0 - floor)).astype(np.int64)
    out: IArr = np.clip(w, 1, levels)
    return out


def neighbour_lists(vectors: npt.NDArray[Any], skip_top: int, candidates: int, neighbours: int, floor: float,
                    levels: int) -> Arrays:
    """``off`` (one slice per word id, len = rows + 1), ``ids`` (sorted inside a slice), ``wt`` (1..levels)."""
    size = len(vectors)
    last = min(candidates, size)
    block = unit_rows(np.asarray(vectors[skip_top:last]))
    counts = np.zeros(size, dtype=np.int64)
    all_ids: list[IArr] = []
    all_wt: list[IArr] = []
    for s in range(0, len(block), CHUNK):
        sims = block[s:s + CHUNK] @ block.T
        sims[np.arange(len(sims)), np.arange(s, s + len(sims))] = -np.inf
        ids, cos = _top(sims, neighbours, floor)
        for r, (cols, c) in enumerate(zip(ids, cos, strict=True)):
            order = np.argsort(cols, kind="stable")
            all_ids.append(cols[order] + skip_top)
            all_wt.append(_weights(c[order], floor, levels))
            counts[skip_top + s + r] = len(cols)
    empty = np.zeros(0, dtype=np.int64)
    ids_out = np.concatenate(all_ids) if all_ids else empty
    wt_out = np.concatenate(all_wt) if all_wt else empty
    return {"off": cumulative(counts), "ids": ids_out.astype(np.int32), "wt": wt_out.astype(np.uint16)}
