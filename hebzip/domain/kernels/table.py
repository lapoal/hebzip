"""The word coding table F(x) - one dense part plus any number of sparse parts.

F(x) = a * cumU[x] + sum over parts k of cum_k[searchsorted_left(ids_k, x)] is the cumulative
weight of all word ids below x. Each predictor contributes its own sparse part; integer sums do not
depend on how the weights are split, so F is the same function the research prototype built from merged
arrays. ``find`` is a plain bisection: F is strictly increasing (every word has a*U[x] >= 1), so
"the largest x with F(x) <= target" has exactly one answer, whatever search finds it.
Layout (no lists of arrays - numba needs flat arrays): part k's ids are ``ids[offs[k]:offs[k+1]]``
and its cumulative weights (one longer, starting with 0) start at ``cums[offs[k] + k]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.kernels.jit import kernel

IArr = npt.NDArray[np.int64]


@kernel
def lower_bound(arr: npt.NDArray[Any], lo: int, hi: int, x: Any) -> int:
    """First index in [lo, hi) with arr[i] >= x (hi when none) - numpy's searchsorted 'left'.

    Used for int64 word ids and for uint64 template keys (numba compiles one copy per type)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@kernel
def table_value(x: int, a: int, cum_u: IArr, ids: IArr, cums: IArr, offs: IArr) -> Any:
    """F(x): the total weight of every id below ``x`` (an int64 scalar)."""
    v = a * cum_u[x]
    for k in range(offs.shape[0] - 1):
        start = offs[k]
        j = lower_bound(ids, start, offs[k + 1], x)
        v += cums[start + k + (j - start)]
    return v


@kernel
def table_find(target: int, a: int, cum_u: IArr, ids: IArr, cums: IArr, offs: IArr) -> int:
    """The largest x with F(x) <= target."""
    lo = 0
    hi = cum_u.shape[0] - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if table_value(mid, a, cum_u, ids, cums, offs) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo
