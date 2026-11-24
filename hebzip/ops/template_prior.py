"""The template model's corpus prior: static context tables (``template_prior``) and start weights
(``template_weights``) - the research prototype's ``tp_tables.py`` and ``tp_train.py`` as recipes.

Static tables: for every template byte of the training documents and each of its 8 bits (MSB first),
the context key = ``finish(context hash, bits so far)`` - with the very hash kernels the coder runs
(``KernelSet.ctx_*``, python engine, applied to whole numpy arrays: one source). Per key the counts
of 0 and 1 (summed over the bit positions, like the research prototype) give a KT probability
``(2*n1 + 1) / (2*n + 2)`` in ``prob_bits`` bits (the template row), stored in the stretch domain
divided by ``static_scale``, plus a confidence level (how many ``count_levels`` the count reaches). Keys
seen fewer than ``min_count`` times are dropped. Layout per input: ``<name>_dkey`` (key deltas, uint32),
``<name>_st``, ``<name>_lvl``.
Start weights: ``domain/template_fit.py`` on the held-out documents. Zero I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import KernelSet
from hebzip.domain.kernels_layout import WCAP
from hebzip.domain.template_cm import CmSpec
from hebzip.domain.template_tables import stretch_table
from hebzip.ops.counts import Arrays, Doc

UArr = npt.NDArray[np.uint64]
IArr = npt.NDArray[np.int64]
BYTE_BITS = 8
INT8 = (-(1 << 7), (1 << 7) - 1)


class Stream:
    """All training documents as one byte array, each byte knowing where its document starts."""

    def __init__(self, docs: list[Doc], slot_end: frozenset[int]) -> None:
        self.buf = np.frombuffer(b"".join(t for t, _ in docs), dtype=np.uint8).astype(np.int64)
        lengths = [len(t) for t, _ in docs]
        firsts = np.cumsum([0] + lengths[:-1]).astype(np.int64)
        self.start = np.repeat(firsts, lengths).astype(np.int64)
        self.slots = np.array([x for _, ids in docs for x in ids] or [0], dtype=np.int64)
        self.idx = np.arange(len(self.buf), dtype=np.int64)
        self.is_end = np.isin(self.buf, sorted(slot_end))


def order_hash(s: Stream, order: int, k: KernelSet, hp: UArr) -> UArr:
    """The order-``order`` context hash of every byte: the previous bytes, oldest first, 0 before the start."""
    h: Any = np.full(len(s.buf), k.ctx_seed(order, hp), dtype=np.uint64)
    for back in range(order, 0, -1):
        j = s.idx - back
        h = k.ctx_roll(h, np.where(j >= s.start, s.buf[np.maximum(j, 0)], 0), hp)
    out: UArr = h
    return out


def word_hash(s: Stream, cap: int, k: KernelSet, hp: UArr, ip: IArr) -> UArr:
    """The "after word w, d bytes ago" hash of every byte (no word yet in the document: w = -1)."""
    n_before = np.concatenate([[0], np.cumsum(s.is_end)[:-1]]).astype(np.int64)
    last_end = np.maximum.accumulate(np.where(s.is_end, s.idx, -1))
    last_end = np.concatenate([[-1], last_end[:-1]]).astype(np.int64)
    has = last_end >= s.start
    wid = np.where(has, s.slots[np.maximum(n_before - 1, 0)], -1)
    dist = np.minimum(np.where(has, s.idx - last_end - 1, s.idx - s.start), cap)
    out: UArr = k.ctx_word(wid, dist, hp, ip)
    return out


def bit_counts(h: UArr, buf: IArr, k: KernelSet, hp: UArr) -> tuple[UArr, IArr, IArr]:
    """Sorted unique keys over all 8 bit positions, with their counts of 0 and of 1."""
    keys, ones, counts = [], [], []
    for j in range(BYTE_BITS):
        c0 = (buf | (1 << BYTE_BITS)) >> (BYTE_BITS - j)
        y = (buf >> (BYTE_BITS - 1 - j)) & 1
        both, c = np.unique((k.ctx_finish(h, c0, hp) << np.uint64(1)) | y.astype(np.uint64), return_counts=True)
        keys.append(both >> np.uint64(1))
        ones.append((both & np.uint64(1)).astype(bool))
        counts.append(c.astype(np.int64))
    key, one, count = np.concatenate(keys), np.concatenate(ones), np.concatenate(counts)
    uk, inv = np.unique(key, return_inverse=True)
    n0 = np.bincount(inv[~one], weights=count[~one], minlength=len(uk)).astype(np.int64)
    n1 = np.bincount(inv[one], weights=count[one], minlength=len(uk)).astype(np.int64)
    return uk, n0, n1


def entries(keys: UArr, n0: IArr, n1: IArr, min_count: int, levels: list[int], stretch: IArr,
            static_scale: int) -> Arrays:
    """Kept keys (count >= ``min_count``) -> ``dkey`` / ``st`` / ``lvl`` arrays. The stretch table has
    one entry per probability, so its length is 2**prob_bits."""
    n = n0 + n1
    keep = n >= min_count
    keys, n1, n = keys[keep], n1[keep], n[keep]
    one = len(stretch)
    p = np.clip(((2 * n1 + 1) * one) // (2 * n + 2), 1, one - 1)
    st = stretch[p] // static_scale
    small = len(st) == 0 or (INT8[0] <= int(st.min()) and int(st.max()) <= INT8[1])
    return {"dkey": np.diff(keys.astype(np.int64), prepend=0).astype(np.uint32),
            "st": st.astype(np.int8 if small else np.int16),
            "lvl": np.searchsorted(np.array(levels, dtype=np.int64), n, side="right").astype(np.uint8)}


def static_prior(docs: list[Doc], spec: CmSpec, slot_end: frozenset[int], k: KernelSet,
                 recipe: tuple[int, list[int], int]) -> Arrays:
    """Every static input's table. ``recipe`` = (min_count, count_levels, static_scale)."""
    min_count, levels, static_scale = recipe
    s = Stream(docs, slot_end)
    hp, ip = spec.kernel_arrays()
    stretch = stretch_table(spec.squash_knots, spec.precision)
    out: Arrays = {}
    for name, order in zip(spec.ctx_names(), list(spec.orders) + [-1], strict=False):
        h = word_hash(s, int(ip[WCAP]), k, hp, ip) if order < 0 else order_hash(s, order, k, hp)
        for part, values in entries(*bit_counts(h, s.buf, k, hp), min_count, levels, stretch, static_scale).items():
            out[f"{name}_{part}"] = values
    return out


def start_weights(spec: CmSpec, value: int) -> IArr:
    """The mixer's weights before fitting: every input of every weight set at ``value``."""
    return np.full(spec.weight_shape(), value, dtype=np.int64)
