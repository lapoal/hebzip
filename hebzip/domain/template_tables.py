"""Pure builders for the template model: squash/stretch from the knot row, static tables from arrays.

Why integer tables from a row of knots: encoder and decoder must compute bit-identical
probabilities, so no exp/log while coding - squash is a piecewise-linear curve through 2**m + 1
integer knots and stretch is its inverse table. The knot values are samples of the logistic
function, made once by ``hebzip.ops.knots`` (formula and rounding written there; a test checks
every row against it). The knots are a config row (``squash_knots``) and so is the
precision (``prob_bits``, ``fix_bits`` - not constants in the kernel); everything
else here follows from them (``Precision``: the one place the derived values are computed). Zero I/O:
the caller hands in the arrays it loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch

IArr = npt.NDArray[np.int64]
UArr = npt.NDArray[np.uint64]


@dataclass(frozen=True)
class Precision:
    """The template model's two precision rows and every value derived from them (one source: the
    kernel reads the derived values from ``ip``, ``kernel_consts`` - it never derives them again)."""

    prob_bits: int
    fix_bits: int

    @property
    def one(self) -> int:
        """2**prob_bits: probabilities are 1 .. one - 1."""
        return 1 << self.prob_bits

    @property
    def span(self) -> int:
        """The stretch (ln-odds) domain is -span..span."""
        return self.one // 2 - 1

    def knot_step_bits(self, knots: tuple[int, ...]) -> int:
        """log2 of the knot spacing: 2**m + 1 knots span 2**prob_bits (``validate_model`` checks the count)."""
        return self.prob_bits - ((len(knots) - 1).bit_length() - 1)

    def kernel_consts(self) -> tuple[int, ...]:
        """The ``ip`` slots PB, FB, P_ONE, P_SPAN, TO_PROB, HALF_FIX (``kernels_layout``), in that order."""
        return (self.prob_bits, self.fix_bits, self.one, self.span, self.fix_bits - self.prob_bits,
                1 << (self.fix_bits - 1))


def squash_table(knots: tuple[int, ...], prec: Precision) -> IArr:
    """The probability of a 1 for every stretch value d = -span..span (entry d + span).

    The curve is the straight line between neighbouring knots. The knots are spread evenly over the
    whole domain -one/2..+one/2: knot i sits at d = i * step - one/2, step = 2**knot_step_bits.
    For a d with ``seg, rest = divmod(d + one/2, step)`` the value is

        k[seg] + floor(((k[seg + 1] - k[seg]) * rest + step/2) / step)

    i.e. the line rounded to the nearest integer (an exact half rounds up), then held inside the
    valid probabilities 1..one-1. Integers only, so encoder and decoder get the same table.
    """
    shift = prec.knot_step_bits(knots)
    step = 1 << shift
    k = np.asarray(knots, dtype=np.int64)
    d = np.arange(-prec.span, prec.span + 1, dtype=np.int64)
    seg, rest = np.divmod(d + prec.one // 2, step)
    line = k[seg] + (((k[seg + 1] - k[seg]) * rest + step // 2) >> shift)
    held: IArr = np.clip(line, 1, prec.one - 1).astype(np.int64)
    return held


def stretch_table(knots: tuple[int, ...], prec: Precision) -> IArr:
    """The way back from a probability p (0..one-1) to the stretch domain: the smallest d whose
    squashed value is at least p; a p above every squashed value gives +span.

    The squash table never falls as d grows, so "the first d reaching p" is a sorted search."""
    curve = squash_table(knots, prec)
    first = np.searchsorted(curve, np.arange(prec.one, dtype=np.int64), side="left")
    return (np.minimum(first, len(curve) - 1) - prec.span).astype(np.int64)


def recip_table(offset2: int, count_limit: int, prec: Precision) -> IArr:
    """1/(n + offset2/2) in ``fix_bits`` fixed point, the adaptive counts' learning rate per count
    n = 0..count_limit (a count never passes ``count_limit``, so that is the whole table)."""
    return np.array([(1 << (prec.fix_bits + 1)) // (2 * n + offset2) for n in range(count_limit + 1)],
                    dtype=np.int64)


@dataclass(frozen=True)
class StaticTables:
    """All static context tables flattened: input i owns ``keys[offs[i]:offs[i+1]]``."""

    keys: UArr
    st16: IArr
    conf: IArr
    offs: IArr


def static_tables(arrays: Mapping[str, npt.NDArray[np.generic]], names: list[str], conf_level: int,
                  st_scale: int) -> StaticTables:
    """Per name: keys = cumsum(dkey) as uint32 (strictly increasing), value = st * st_scale."""
    keys, st16, conf, offs = [], [], [], [0]
    for name in names:
        try:
            k = np.cumsum(arrays[f"{name}_dkey"].astype(np.int64)).astype(np.uint32).astype(np.uint64)
            st = arrays[f"{name}_st"].astype(np.int64) * st_scale
            lvl = arrays[f"{name}_lvl"].astype(np.int64)
        except KeyError as error:
            raise TableMismatch(f"טבלת התבנית חסרה את הקלט {name}") from error
        if len(k) > 1 and not bool(np.all(k[1:] > k[:-1])):
            # a lookup by binary search equals the research prototype's dict only on strictly sorted keys
            raise TableMismatch(f"טבלת התבנית {name}: המפתחות אינם ממוינים חד-חד-ערכית")
        keys.append(k)
        st16.append(st)
        conf.append((lvl >= conf_level).astype(np.int64))
        offs.append(offs[-1] + len(k))
    return StaticTables(np.concatenate(keys), np.concatenate(st16), np.concatenate(conf),
                        np.array(offs, dtype=np.int64))
