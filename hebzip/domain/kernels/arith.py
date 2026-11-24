"""Exact 128-bit helpers on unsigned 64-bit integers.

Why: the range coder multiplies a 64-bit interval width by a 62-bit frequency. Python ints are
unbounded, numba's are 64-bit - so the product is built from 32-bit halves and divided bit by bit.
Every intermediate value stays below 2**64 by construction, so the SAME source gives the same
numbers as plain Python (numpy uint64 scalars) and compiled (numba uint64). Never mix a uint64 with
a signed int here: numpy and numba both turn that into a float64.
"""

from __future__ import annotations

import numpy as np

from hebzip.domain.kernels.jit import kernel
from hebzip.domain.kernels_layout import FULL_INT, HALF_INT, M32_INT, REG_BITS

U = np.uint64
FULL = U(FULL_INT)
M32 = U(M32_INT)
S32 = U(REG_BITS // 2)
TOP = U(REG_BITS - 1)
ONE = U(1)
ZERO = U(0)
LOW63 = U(HALF_INT - 1)


@kernel
def mul_wide(a: np.uint64, b: np.uint64) -> tuple[np.uint64, np.uint64]:
    """a * b as (high 64 bits, low 64 bits)."""
    al = a & M32
    ah = a >> S32
    bl = b & M32
    bh = b >> S32
    ll = al * bl
    lh = al * bh
    hl = ah * bl
    hh = ah * bh
    cross = (ll >> S32) + (lh & M32) + (hl & M32)
    lo = ((cross & M32) << S32) | (ll & M32)
    hi = hh + (lh >> S32) + (hl >> S32) + (cross >> S32)
    return hi, lo


@kernel
def add_wide(hi: np.uint64, lo: np.uint64, c: np.uint64) -> tuple[np.uint64, np.uint64]:
    """(hi, lo) + c, carrying into ``hi`` without any wrap-around."""
    if lo > FULL - c:
        return hi + ONE, lo - (FULL - c) - ONE
    return hi, lo + c


@kernel
def div_wide(hi: np.uint64, lo: np.uint64, d: np.uint64) -> np.uint64:
    """floor((hi * 2**64 + lo) / d) for 1 <= d and hi < d (so the quotient fits in 64 bits).

    Restoring division, one bit at a time. When the remainder's top bit is set, doubling it
    would need 65 bits: then (2*rem + bit) - d is computed as (low 64 bits) + (2**64 - d)."""
    q = ZERO
    rem = hi
    for i in range(REG_BITS - 1, -1, -1):
        carry = rem >> TOP
        doubled = ((rem & LOW63) << ONE) | ((lo >> U(i)) & ONE)
        if carry != ZERO:
            rem = doubled + ((FULL - d) + ONE)
            q = q | (ONE << U(i))
        elif doubled >= d:
            rem = doubled - d
            q = q | (ONE << U(i))
        else:
            rem = doubled
    return q


@kernel
def scale_down(width_m1: np.uint64, x: np.uint64, total: np.uint64) -> np.uint64:
    """floor((width_m1 + 1) * x / total) for x < total <= 2**63 - the coder's r*x//T."""
    hi, lo = mul_wide(width_m1, x)
    hi, lo = add_wide(hi, lo, x)
    return div_wide(hi, lo, total)
