"""The template model's byte step: context mixing over static + adaptive orders.

Per template bit (MSB first) the inputs, in the stretch (ln-odds) domain, are: the corpus prior of
each static context (orders + "after word w"), the text's own adaptive estimate of each adaptive
context, and a bias. A weight set chosen by the longest confident static order (and "word context
confident") mixes them; the mixer and the adaptive counts learn after every bit. Integers only,
so sender and receiver compute the same probabilities on any machine (and in both engines).

Everything numeric comes from config rows through two small arrays: ``hp`` (uint64 hash constants)
and ``ip`` (int64 learning constants + the precision rows ``prob_bits``/``fix_bits`` and the values
derived from them) - see ``domain/kernels_layout.py`` for the slot names. Hash
values never exceed 2**64: a 32-bit hash times a 32-bit constant, masked (the research prototype's maths).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.domain.kernels.arith import M32, U
from hebzip.domain.kernels.jit import kernel
from hebzip.domain.kernels.rc import dec_bit, enc_bit
from hebzip.domain.kernels.table import lower_bound
from hebzip.domain.kernels_layout import (
    AD_COUNT,
    AD_INPUT_BITS,
    AD_MAX,
    AD_MIN,
    AD_START,
    AD_STATUS,
    BIAS,
    C0_MUL,
    FB,
    FIN_MUL,
    FIN_SHIFT1,
    FIN_SHIFT2,
    H_MUL,
    HALF_FIX,
    LIMIT,
    LR,
    M32_INT,
    P_ONE,
    P_SPAN,
    PB,
    SEED_ADD,
    SEED_MUL,
    ST_WEIGHT,
    TO_PROB,
    WCAP,
    WMAX,
    WSEED,
)

UArr = npt.NDArray[np.uint64]
IArr = npt.NDArray[np.int64]
BArr = npt.NDArray[np.uint8]
MARK = U(1 << 40)
"""Adaptive-table key layout ``finish << AD_INPUT_BITS | input | MARK`` (never 0 = empty slot)."""
INPUT_SHIFT = U(AD_INPUT_BITS)
COIN_CUT = 1
COIN_BITS = 1
"""The plate's fair coin: cut 1 of 2**1 (the very split ``cut = 2**(pb-1)`` gives at any ``pb``)."""


@kernel
def seed(k: int, hp: UArr) -> Any:
    """Start hash of a context of order ``k``."""
    return (U(k) * hp[SEED_MUL] + hp[SEED_ADD]) & M32


@kernel
def roll(h: np.uint64, b: int, hp: UArr) -> Any:
    """Fold one more context byte (oldest first) into a hash."""
    return (h * hp[H_MUL] + U(b) + U(1)) & M32


@kernel
def finish(h: np.uint64, c0: int, hp: UArr) -> Any:
    """Context hash + the bits of the byte so far (1..255, leading 1) -> 32-bit table key."""
    x = (h + U(c0) * hp[C0_MUL]) & M32
    x ^= x >> hp[FIN_SHIFT1]
    x = (x * hp[FIN_MUL]) & M32
    x ^= x >> hp[FIN_SHIFT2]
    return x


@kernel
def word_ctx(wid: Any, dist: Any, hp: UArr, ip: IArr) -> Any:
    """The "after word ``wid``, ``dist`` bytes ago" context hash (``dist`` already capped by the caller).

    One source: the coder calls it per byte, the template-prior builder (``ops``) on whole arrays."""
    return roll(roll(seed(ip[WSEED], hp), (wid + 1) & M32_INT, hp), dist, hp)


@kernel
def byte_hashes(hb: UArr, hkind: IArr, hist: IArr, ms: IArr, hp: UArr, ip: IArr) -> None:
    """Every context hash for the next byte: order k = last k bytes; -1 = (last word, distance)."""
    kmax = hist.shape[0]
    for j in range(hkind.shape[0]):
        k = hkind[j]
        if k < 0:
            hb[j] = word_ctx(ms[0], min(ms[1], ip[WCAP]), hp, ip)
        else:
            x = seed(k, hp)
            for t in range(kmax - k, kmax):
                x = roll(x, hist[t], hp)
            hb[j] = x


@kernel
def slot_of(akeys: UArr, key: np.uint64) -> Any:
    """Open addressing: the slot holding ``key``, or the empty slot where it belongs."""
    mask = akeys.shape[0] - 1
    i = np.int64(key & U(mask))
    while akeys[i] != U(0) and akeys[i] != key:
        i = (i + 1) & mask
    return i


@kernel
def static_inputs(c0: int, hb: UArr, s_slot: IArr, s_word: IArr, skeys: UArr, sst: IArr, sconf: IArr,
                  soff: IArr, xs: IArr, hp: UArr) -> int:
    """Static inputs into ``xs``; returns the weight-set selector (longest confident order, word bit)."""
    sel = 0
    for ii in range(s_slot.shape[0]):
        key = finish(hb[s_slot[ii]], c0, hp)
        end = soff[ii + 1]
        idx = lower_bound(skeys, soff[ii], end, key)
        xs[ii] = 0
        if idx < end and skeys[idx] == key:
            xs[ii] = sst[idx]
            if sconf[idx]:
                if s_word[ii]:
                    sel = sel | 1
                else:
                    sel = (sel & 1) | ((ii + 1) << 1)
    return sel


@kernel
def adaptive_inputs(c0: int, hb: UArr, a_slot: IArr, akey: UArr, akeys: UArr, ap: IArr, xs: IArr,
                    first: int, stretch: IArr, hp: UArr, ip: IArr) -> None:
    """The text's own estimates into ``xs[first:]`` (0 for a context never seen)."""
    for jj in range(a_slot.shape[0]):
        key = (finish(hb[a_slot[jj]], c0, hp) << INPUT_SHIFT) | U(jj) | MARK
        akey[jj] = key
        slot = slot_of(akeys, key)
        xs[first + jj] = stretch[ap[slot] >> ip[TO_PROB]] if akeys[slot] == key else 0


@kernel
def mix(w_row: IArr, xs: IArr, sqt: IArr, ip: IArr) -> Any:
    """Fixed-point dot product -> squashed ``prob_bits`` probability of a 1, clamped to [1, 2**pb - 1]."""
    dot = 0
    for i in range(xs.shape[0]):
        dot += w_row[i] * xs[i]
    d = dot >> ip[FB]
    if d > ip[P_SPAN]:
        return ip[P_ONE] - 1
    if d < -ip[P_SPAN]:
        return 1
    return sqt[d + ip[P_SPAN]]


@kernel
def learn(w_row: IArr, xs: IArr, y: int, p: int, ip: IArr, ameta: IArr) -> None:
    """Mixer gradient step; a weight beyond ``WMAX`` stops both engines alike (status)."""
    err = ((y << ip[PB]) - p) * ip[LR]
    for i in range(xs.shape[0]):
        w = w_row[i] + ((xs[i] * err + ip[HALF_FIX]) >> ip[FB])
        if w > ip[WMAX] or w < -ip[WMAX]:
            ameta[AD_STATUS] = ST_WEIGHT
        w_row[i] = w


@kernel
def adapt(y: int, akey: UArr, akeys: UArr, ap: IArr, an: IArr, ameta: IArr, recip: IArr, ip: IArr) -> None:
    """Update (or create) the adaptive count of every adaptive context."""
    for jj in range(akey.shape[0]):
        key = akey[jj]
        slot = slot_of(akeys, key)
        if akeys[slot] != key:
            akeys[slot] = key
            ap[slot] = ip[AD_START]
            an[slot] = 0
            ameta[AD_COUNT] += 1
        e0 = ap[slot] + ((((y << ip[FB]) - ap[slot]) * recip[an[slot]]) >> ip[FB])
        ap[slot] = min(max(e0, ip[AD_MIN]), ip[AD_MAX])
        an[slot] = min(an[slot] + 1, ip[LIMIT])


@kernel
def cm_byte(enc: int, byte: int, cu: UArr, ci: IArr, bits: BArr, model: Any, tabs: Any, ad: Any) -> int:
    """Code one template byte (encoder: ``byte``; decoder: returns the decoded byte)."""
    hp, ip, hkind, hist, ms, w, xs, hb, stretch, sqt, recip = model
    s_slot, s_word, skeys, sst, sconf, soff, a_slot = tabs
    akey, akeys, ap, an, ameta = ad
    byte_hashes(hb, hkind, hist, ms, hp, ip)
    nsta = s_slot.shape[0]
    c0 = 1
    for jbit in range(8):
        sel = static_inputs(c0, hb, s_slot, s_word, skeys, sst, sconf, soff, xs, hp)
        adaptive_inputs(c0, hb, a_slot, akey, akeys, ap, xs, nsta, stretch, hp, ip)
        xs[xs.shape[0] - 1] = ip[BIAS]
        p = mix(w[sel], xs, sqt, ip)
        if enc:
            y = (byte >> (7 - jbit)) & 1
            enc_bit(cu, ci, bits, ip[P_ONE] - p, y, ip[PB])
        else:
            y = dec_bit(cu, ci, bits, ip[P_ONE] - p, ip[PB])
        learn(w[sel], xs, y, p, ip, ameta)
        adapt(y, akey, akeys, ap, an, ameta, recip, ip)
        c0 = (c0 << 1) | y
    b = c0 & 255
    kmax = hist.shape[0]
    for t in range(kmax - 1):
        hist[t] = hist[t + 1]
    if kmax:
        hist[kmax - 1] = b
    ms[1] += 1
    return b


@kernel
def rehash(old_keys: UArr, old_p: IArr, old_n: IArr, keys: UArr, p: IArr, n: IArr) -> None:
    """Move every adaptive entry into a bigger table (values unchanged, only slots move)."""
    for i in range(old_keys.shape[0]):
        key = old_keys[i]
        if key != U(0):
            slot = slot_of(keys, key)
            keys[slot] = key
            p[slot] = old_p[i]
            n[slot] = old_n[i]


@kernel
def flat_byte(enc: int, byte: int, cu: UArr, ci: IArr, bits: BArr) -> int:
    """The template plate: every bit at probability 1/2 (8 bits per byte, learns nothing)."""
    c0 = 1
    for jbit in range(8):
        if enc:
            y = (byte >> (7 - jbit)) & 1
            enc_bit(cu, ci, bits, COIN_CUT, y, COIN_BITS)
        else:
            y = dec_bit(cu, ci, bits, COIN_CUT, COIN_BITS)
        c0 = (c0 << 1) | y
    return c0 & 255
