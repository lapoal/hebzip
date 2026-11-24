"""Static letter chains: corpus letter statistics, interpolated over context orders.

Symbols: 0 = word end, 1..L = letters. A context of order k is the last k symbols packed as a
base-(L+2) number, digit = symbol + 1 (digit 0 = nothing: before the text start). Two contexts:
  cross   the last k symbols of the text (earlier words' letters and ends included)
  within  the same, cut at the start of the current word (the end digit marks the cut)
Table (``letters_v15``-style ``.npz``): ``n0`` = symbol counts; per order k of a context family f
(``x`` = cross, ``w`` = within) ``f{k}_ck`` = sorted ``key << symbol_bits | symbol`` and ``f{k}_cn``
their counts. Prediction (the research prototype's Witten-Bell-like chain, ``ls_model.LModel.chain``):
  p_-1(c) = (n0[c] + add_count) / (N0 + add_count * S)
  p_k(c)  = (d_k * beta * p_{k-1}(c) + n_k(c)) / (N_k + d_k * beta)   while order k has the context
written exactly as the research prototype's float expressions, in its order. The memo is a pure
cache (a result depends only on the key of the highest order); its size limit changes no bit. The
memo also keeps log P per symbol once asked (``logs``) - the same ``log(p[c])``, computed once
per context instead of once per visit.
"""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np
import numpy.typing as npt

IArr = npt.NDArray[np.int64]
Ints = npt.NDArray[Any]
"""A chain array of any integer width - pinned, narrowed or a read-only map: ``counts`` turns
every value into a Python ``int`` (``tolist``) before any arithmetic, so the width never reaches a number."""
MEMO_LIMIT = 400_000
"""Cleared when larger (memory only)."""
FAMILY = {"cross": "x", "within": "w"}
"""Context word -> the letter of its arrays in the table (``x1_ck`` ...)."""
WITHIN = {"cross": False, "within": True}
"""Context word -> is the context cut at the start of the current word (a table, not ``== "within"``)."""


def keys_from_hist(hist: list[int], orders: int, within: bool, base: int) -> list[int]:
    """The packed context keys of orders 1..``orders`` after ``hist`` (oldest first, -1 = nothing)."""
    out = []
    key = 0
    mul = 1
    n = len(hist)
    dist = n + 1
    if within:
        dist = 1
        while dist <= n and hist[n - dist] > 0:
            dist += 1
    for j in range(1, orders + 1):
        digit = hist[n - j] + 1 if j <= n else 0
        if within:
            digit = digit if j < dist else (1 if j == dist else 0)
        key += digit * mul
        mul *= base
        out.append(key)
    return out


def _find(ck: Ints, top: int, value: int) -> int:
    """``ck.searchsorted(value)`` with ``value`` in the array's OWN type.

    Why: searching a narrowed (int32 ...) array with a Python int makes numpy cast the WHOLE array to int64
    on every call - 18 ms instead of 2 us on 2M keys, max became 4x slower on the mapped tables. A value
    above the type's largest is above every key (keys are never negative), so it lands at the end."""
    if value > top:
        return len(ck)
    return int(ck.searchsorted(ck.dtype.type(value)))


class LetterChain:
    """One static chain: a context family, its orders 1..K, beta and the order -1 floor."""

    def __init__(self, n0: IArr, entries: list[tuple[Ints, Ints]], within: bool, beta: float, add_count: float,
                 symbol_bits: int) -> None:
        counts = n0.tolist()
        total = 0
        for v in counts:
            total += v
        self.symbols = len(counts)
        self.p0 = [(counts[c] + add_count) / (total + add_count * self.symbols) for c in range(self.symbols)]
        self.entries = entries
        self.tops = [int(np.iinfo(ck.dtype).max) for ck, _ in entries]
        self.orders = len(entries)
        self.within = within
        self.beta = beta
        self.shift = symbol_bits
        self.mask = (1 << symbol_bits) - 1
        self.base = self.symbols + 1
        self.memo: dict[int, list[list[float] | None]] = {}

    def counts(self, k: int, key: int) -> list[tuple[int, int]] | None:
        """(symbol, count) pairs seen after the order-k context ``key``, or ``None``."""
        ck, cn = self.entries[k - 1]
        top = self.tops[k - 1]
        lo = _find(ck, top, key << self.shift)
        hi = _find(ck, top, (key + 1) << self.shift)
        if lo == hi:
            return None
        return [(a & self.mask, b) for a, b in zip(ck[lo:hi].tolist(), cn[lo:hi].tolist(), strict=True)]

    def probs(self, hist: list[int]) -> list[float]:
        """P(next symbol) after ``hist``."""
        p = self._entry(hist)[0]
        assert p is not None
        return p

    def logs(self, hist: list[int]) -> list[float]:
        """log P(next symbol) after ``hist``, per symbol (kept in the memo with the probabilities)."""
        entry = self._entry(hist)
        lp = entry[1]
        if lp is None:
            p = entry[0]
            assert p is not None
            lp = entry[1] = [log(v) for v in p]
        return lp

    def _entry(self, hist: list[int]) -> list[list[float] | None]:
        """The memo cell [P, log P or None] of the context ``hist``."""
        keys = keys_from_hist(hist, self.orders, self.within, self.base)
        top = keys[-1]
        entry = self.memo.get(top)
        if entry is not None:
            return entry
        p = self.p0
        for k in range(1, self.orders + 1):
            cnt = self.counts(k, keys[k - 1])
            if cnt is None:
                break
            big_n = 0
            for _, n in cnt:
                big_n += n
            db = len(cnt) * self.beta
            den = big_n + db
            q = [db * v / den for v in p]
            for c, n in cnt:
                q[c] += n / den
            p = q
        if len(self.memo) > MEMO_LIMIT:
            self.memo.clear()
        entry = self.memo[top] = [p, None]
        return entry
