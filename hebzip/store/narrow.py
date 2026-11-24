"""Narrow integer arrays for the derived table copy - never by cutting a value.

Why derived from the values and not a config row per array: the width an array needs is a fact of its
values, and the values are pinned by the table's sha256; a row per array (80+ of them) would be a second
ledger of the same fact that could disagree with it (the one-ledger-per-fact rule). What someone may want otherwise
is WHICH widths are allowed - that is the row ``[hebzip].narrow_bits`` (e.g. a platform without 8-bit
loads gives ``[16, 32, 64]``). The chosen width is recorded in the manifest next to the copy.
Rules: the narrowest allowed width of the SAME signedness that holds every value (a uint32 table stays
unsigned, an int64 one stays signed - so no drawer ever meets a sign it did not meet before); a value
that fits no allowed width is ``NarrowingOverflow``; the narrowed array is compared value by value with
the source before it is returned. Non-integer arrays (the template's string list) are kept as they are.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import NarrowingOverflow

TYPE_NAMES = {"i": "int", "u": "uint"}
"""Integer kind -> numpy type name (``int16``, ``uint32`` ...); any other kind is not narrowed."""


def narrowest(array: npt.NDArray[np.generic], bits: Sequence[int]) -> np.dtype[np.generic]:
    """The narrowest allowed dtype of the array's signedness that holds all its values."""
    name = TYPE_NAMES.get(array.dtype.kind)
    if name is None:
        return array.dtype
    low, high = (int(array.min()), int(array.max())) if array.size else (0, 0)
    for width in sorted(bits):
        dtype: np.dtype[np.generic] = np.dtype(f"{name}{width}")
        info = np.iinfo(dtype)  # type: ignore[type-var]
        if info.min <= low and high <= info.max:
            return dtype.newbyteorder("<")
    raise NarrowingOverflow(f"ערך {low if low < 0 else high} לא נכנס לאף רוחב מותר ({list(bits)} ביט, {name})")


def narrow(array: npt.NDArray[np.generic], bits: Sequence[int]) -> npt.NDArray[np.generic]:
    """The array in its narrowest allowed width (C order, little endian), proven equal to the source."""
    out = np.ascontiguousarray(array, dtype=narrowest(array, bits))
    if out.shape != array.shape or not np.array_equal(out, array):
        raise NarrowingOverflow(f"צמצום הרוחב שינה ערך (dtype {array.dtype} -> {out.dtype})")
    return out
