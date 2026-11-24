"""One word's coding table: dense base part + sparse parts, coded through the kernels (plain code).

Why the total is checked: the coder's precision needs every table total <= 2**62 (the research prototype
asserted it). With the configured scale (2**60) and weights that sum to 1 this always holds; a
config that breaks it refuses to encode (``DrawerUnavailable``) and a package that would need it is
damaged (``DamagedStream``) - never a silently wrong interval.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import DamagedStream, DrawerUnavailable
from hebzip.domain.contracts import BasePriorPort, CoderPort, KernelSet, Sparse

IArr = npt.NDArray[np.int64]
MAX_TOTAL = 1 << 62


class WordTable:
    """F(x) = a * base.cum[x] + sum of the sparse parts below x."""

    def __init__(self, a: int, base: BasePriorPort, parts: list[Sparse], kernels: KernelSet) -> None:
        self.a = a
        self.base = base
        self.parts = parts
        self.cum_u = base.cum
        self.k = kernels
        self.total = a * base.total + sum(int(p.cum[-1]) for p in parts)
        self._flat: tuple[IArr, IArr, IArr] | None = None

    def flat(self) -> tuple[IArr, IArr, IArr]:
        """The parts as the kernels read them: (ids, cumulative weights, part offsets). Built on first use
        (the ``spelled`` coder reads ``parts`` and never needs this copy)."""
        if self._flat is None:
            empty = np.zeros(0, dtype=np.int64)
            parts = self.parts
            ids = np.concatenate([p.ids.astype(np.int64) for p in parts]) if parts else empty
            cums = np.concatenate([p.cum for p in parts]) if parts else empty
            offs = np.cumsum(np.array([0] + [len(p.ids) for p in parts], dtype=np.int64))
            self._flat = (ids, cums, offs)
        return self._flat

    def value(self, x: int) -> int:
        """F(x)."""
        ids, cums, offs = self.flat()
        return int(self.k.table_value(x, self.a, self.cum_u, ids, cums, offs))

    def code(self, coder: CoderPort, x: int) -> int:
        """Encode word ``x`` / decode one -> the word id."""
        cu, ci, bits = coder.arrays()
        if coder.encoding:
            if self.total > MAX_TOTAL:
                raise DrawerUnavailable("word_model: table total above 2**62")
            lo, hi = self.value(x), self.value(x + 1)
            if hi <= lo:
                raise DrawerUnavailable("word_model: a word with no room in the table")
            self.k.enc_symbol(cu, ci, bits, lo, hi, self.total)
            return x
        if self.total > MAX_TOTAL:
            raise DamagedStream("החבילה פגומה: טבלת מילים גדולה מדי")
        target = int(self.k.dec_target(cu, self.total))
        ids, cums, offs = self.flat()
        x = int(self.k.table_find(target, self.a, self.cum_u, ids, cums, offs))
        self.k.dec_symbol(cu, ci, bits, self.value(x), self.value(x + 1), self.total)
        return x
