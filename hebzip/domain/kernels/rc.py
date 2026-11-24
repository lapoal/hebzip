"""The range coder (Witten-Neal-Cleary, 64-bit registers) - plain code, no alternative (the change-without-code rule).

State lives in two small arrays so the template kernel and the word coder share ONE coder:
``cu`` (uint64) = [low, high, value]; ``ci`` (int64) = [pending, n_out, pos, n_bits, phantom_max,
status]. The encoder appends one bit per array cell to ``bits``; the decoder reads ``bits`` (the
package body unpacked) and reads zeros past its end.

Why ``phantom_max`` (design decision, the "unbounded decoder" defect): past the end of the body a
decoder reads zero bits forever, so a damaged package that declares a long template made the
decoder run for hours. A valid stream never reads more than ``63 + pending_at_finish`` bits past its
end; after ``phantom_max`` such bits the decoder sets ``status`` and the caller raises a
typed error. Work is then bounded by the package size.
Why the interval check (status 2): for a valid stream low <= value <= high always holds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.kernels.arith import FULL, ONE, ZERO, U, add_wide, div_wide, mul_wide, scale_down
from hebzip.domain.kernels.jit import kernel
from hebzip.domain.kernels_layout import (
    HALF_INT,
    NBITS,
    NOUT,
    PEND,
    PMAX,
    POS,
    QTR_INT,
    REG_BITS,
    ST_INTERVAL,
    ST_PHANTOM,
    STATUS,
)

HALF = U(HALF_INT)
QTR = U(QTR_INT)

UArr = npt.NDArray[np.uint64]
IArr = npt.NDArray[np.int64]
BArr = npt.NDArray[np.uint8]


@kernel
def emit(ci: IArr, bits: BArr, bit: int) -> None:
    """Append ``bit`` and then every pending opposite bit."""
    n = ci[NOUT]
    bits[n] = bit
    n += 1
    for _ in range(ci[PEND]):
        bits[n] = 1 - bit
        n += 1
    ci[PEND] = 0
    ci[NOUT] = n


@kernel
def enc_rescale(cu: UArr, ci: IArr, bits: BArr) -> None:
    """Shift out every settled leading bit of the encoder interval."""
    low = cu[0]
    high = cu[1]
    while True:
        if high < HALF:
            emit(ci, bits, 0)
        elif low >= HALF:
            emit(ci, bits, 1)
            low -= HALF
            high -= HALF
        elif low >= QTR and high < HALF + QTR:
            ci[PEND] += 1
            low -= QTR
            high -= QTR
        else:
            break
        low = low << ONE
        high = (high << ONE) | ONE
    cu[0] = low
    cu[1] = high


@kernel
def next_bit(ci: IArr, bits: BArr) -> Any:
    """The next body bit, or 0 past the end (counted against ``phantom_max``)."""
    pos = ci[POS]
    ci[POS] = pos + 1
    if pos < ci[NBITS]:
        return U(bits[pos])
    if pos - ci[NBITS] >= ci[PMAX]:
        ci[STATUS] = ST_PHANTOM
    return ZERO


@kernel
def dec_rescale(cu: UArr, ci: IArr, bits: BArr) -> None:
    """The decoder twin of ``enc_rescale``: the same shifts, reading one bit per shift."""
    low = cu[0]
    high = cu[1]
    value = cu[2]
    if value < low or value > high:
        ci[STATUS] = ST_INTERVAL
        return
    while True:
        if high < HALF:
            pass
        elif low >= HALF:
            low -= HALF
            high -= HALF
            value -= HALF
        elif low >= QTR and high < HALF + QTR:
            low -= QTR
            high -= QTR
            value -= QTR
        else:
            break
        low = low << ONE
        high = (high << ONE) | ONE
        value = (value << ONE) | next_bit(ci, bits)
    cu[0] = low
    cu[1] = high
    cu[2] = value


@kernel
def dec_start(cu: UArr, ci: IArr, bits: BArr) -> None:
    """Fill ``value`` with the first 64 bits."""
    value = ZERO
    for _ in range(REG_BITS):
        value = (value << ONE) | next_bit(ci, bits)
    cu[0] = ZERO
    cu[1] = FULL
    cu[2] = value


@kernel
def split_point(cu: UArr, cut: int, pb: int) -> Any:
    """low + r*cut//2**pb with r = high-low+1 (up to 2**64): q*cut + ((rem+1)*cut >> pb), exact.

    ``pb`` = the probability bits of the caller (the template row ``prob_bits``)."""
    low = cu[0]
    width_m1 = cu[1] - low
    c = U(cut)
    s = U(pb)
    return low + (width_m1 >> s) * c + ((((width_m1 & ((ONE << s) - ONE)) + ONE) * c) >> s)


@kernel
def enc_bit(cu: UArr, ci: IArr, bits: BArr, cut: int, y: int, pb: int) -> None:
    """Code one binary decision: P(y=1) = (2**pb - cut) / 2**pb."""
    s = split_point(cu, cut, pb)
    if y:
        cu[0] = s
    else:
        cu[1] = s - ONE
    enc_rescale(cu, ci, bits)


@kernel
def dec_bit(cu: UArr, ci: IArr, bits: BArr, cut: int, pb: int) -> int:
    """Decode one binary decision (value >= split <=> y = 1, exact for every stream)."""
    s = split_point(cu, cut, pb)
    y = 0
    if cu[2] >= s:
        y = 1
        cu[0] = s
    else:
        cu[1] = s - ONE
    dec_rescale(cu, ci, bits)
    return y


@kernel
def narrow(cu: UArr, lo: int, hi: int, total: int) -> None:
    """New interval for symbol [lo, hi) of ``total`` (both bounds from the OLD low)."""
    low = cu[0]
    width_m1 = cu[1] - low
    t = U(total)
    if hi == total:
        new_high = cu[1]
    else:
        new_high = low + scale_down(width_m1, U(hi), t) - ONE
    cu[0] = low + scale_down(width_m1, U(lo), t)
    cu[1] = new_high


@kernel
def enc_symbol(cu: UArr, ci: IArr, bits: BArr, lo: int, hi: int, total: int) -> None:
    """Code a symbol with cumulative range [lo, hi) of ``total``."""
    narrow(cu, lo, hi, total)
    enc_rescale(cu, ci, bits)


@kernel
def dec_target(cu: UArr, total: int) -> np.int64:
    """((value-low+1)*total - 1) // r, exact for r = high-low+1 up to 2**64.

    Returned as int64 (it is < total <= 2**62): numba keeps ``int(uint64)`` unsigned, and a later
    compare with the int64 table values would silently become a float compare."""
    width_m1 = cu[1] - cu[0]
    t = U(total)
    hi, lo = mul_wide(cu[2] - cu[0], t)
    hi, lo = add_wide(hi, lo, t - ONE)
    if width_m1 == FULL:
        return np.int64(hi)
    return np.int64(div_wide(hi, lo, width_m1 + ONE))


@kernel
def dec_symbol(cu: UArr, ci: IArr, bits: BArr, lo: int, hi: int, total: int) -> None:
    """Consume the symbol [lo, hi) found from ``dec_target``."""
    narrow(cu, lo, hi, total)
    dec_rescale(cu, ci, bits)
