"""Template-model drawers: ``cm`` (context mixing with a corpus prior) and the plate ``flat``.

Why the context orders are rows, not drawers (design decision): the orders, the adaptive orders,
the "after word" context and every learning constant come from ``[modes.<m>.template]``; the math
that combines them is one kernel (``kernels/cm.py``). A new set of orders is a new row + a table
built with those orders - no code. The plate codes every bit at 1/2: always works, learns nothing.
Per message the model starts from the shipped weights (both sides must start identical).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import DamagedStream, DrawerUnavailable
from hebzip.domain.contracts import CoderPort, KernelSet
from hebzip.domain.kernels_layout import AD_STATUS, HP_SIZE, IP_SIZE, LEARN_SIZE, LIMIT
from hebzip.domain.template_tables import Precision, StaticTables, recip_table, squash_table, stretch_table

IArr = npt.NDArray[np.int64]
WORD = "w"
START_SLOTS = 1 << 12
BYTE_BITS_RESERVE = 128


@dataclass(frozen=True)
class CmSpec:
    """The numbers of one ``cm`` template model (copied from its config rows by the services)."""

    orders: tuple[int, ...]
    adaptive: tuple[int, ...]
    word_context: bool
    hash_consts: tuple[int, ...]
    learn_consts: tuple[int, ...]
    squash_knots: tuple[int, ...]
    recip_offset2: int
    precision: Precision

    def kernel_arrays(self) -> tuple[npt.NDArray[np.uint64], IArr]:
        """``hp`` (hash constants) and ``ip`` (learning rows, then the precision rows and what follows
        from them - ``kernels_layout``) as the kernels read them."""
        if len(self.hash_consts) != HP_SIZE or len(self.learn_consts) != LEARN_SIZE:
            raise DrawerUnavailable("template_model: wrong number of constants")
        ip = self.learn_consts + self.precision.kernel_consts()
        assert len(ip) == IP_SIZE
        return np.array(self.hash_consts, dtype=np.uint64), np.array(ip, dtype=np.int64)

    def ctx_names(self) -> list[str]:
        """Static inputs in order: the orders, then the word context."""
        return [str(o) for o in self.orders] + ([WORD] if self.word_context else [])

    def ad_names(self) -> list[str]:
        """Adaptive inputs in order."""
        return [str(o) for o in self.adaptive] + ([WORD] if self.word_context else [])

    def weight_shape(self) -> tuple[int, int]:
        """(weight sets, inputs): a set per (longest confident static order or none) x (word context
        confident), an input per static and adaptive context plus the bias."""
        return 2 * (len(self.orders) + 1), len(self.ctx_names()) + len(self.ad_names()) + 1


class CmTemplateModel:
    """Static + adaptive contexts, gated mixer - the template coder of mode fast."""

    name = "cm"

    def __init__(self, spec: CmSpec, static: StaticTables, weights: IArr, kernels: KernelSet) -> None:
        self.k = kernels
        hnames = list(dict.fromkeys(spec.ctx_names() + spec.ad_names()))
        self.hkind = np.array([-1 if n == WORD else int(n) for n in hnames], dtype=np.int64)
        self.tabs = (np.array([hnames.index(n) for n in spec.ctx_names()], dtype=np.int64),
                     np.array([1 if n == WORD else 0 for n in spec.ctx_names()], dtype=np.int64),
                     static.keys, static.st16, static.conf, static.offs,
                     np.array([hnames.index(n) for n in spec.ad_names()], dtype=np.int64))
        self.hp, self.ip = spec.kernel_arrays()
        self.w0 = weights.astype(np.int64)
        n_in = spec.weight_shape()[1]
        if self.w0.shape != spec.weight_shape():
            raise DrawerUnavailable(f"template_model: weights shape {self.w0.shape} does not fit the orders")
        self.n_in, self.n_ad = n_in, len(spec.ad_names())
        self.kmax = max(spec.orders + spec.adaptive)
        prec = spec.precision
        self.tables = (stretch_table(spec.squash_knots, prec), squash_table(spec.squash_knots, prec),
                       recip_table(spec.recip_offset2, spec.learn_consts[LIMIT], prec))
        self.start(0)

    def start(self, n: int) -> None:
        """A new message: shipped weights, empty history, no word yet, empty notebook."""
        self.w = self.w0.copy()
        self.hist = np.zeros(self.kmax, dtype=np.int64)
        self.ms = np.array([-1, 0], dtype=np.int64)
        self.xs = np.zeros(self.n_in, dtype=np.int64)
        self.hb = np.zeros(self.hkind.shape[0], dtype=np.uint64)
        self.akey = np.zeros(self.n_ad, dtype=np.uint64)
        self.ad = self._notebook(START_SLOTS)

    def carry_weights(self, n: int) -> None:
        """A new text that keeps the mixer weights learned so far - only for FITTING the start weights
        (the research prototype's ``reset(keep_weights=True)``, ``domain/template_fit.py``); coding never does this."""
        w = self.w
        self.start(n)
        self.w = w

    def weights(self) -> IArr:
        """The mixer weights as they are now (one row per weight set)."""
        return self.w.copy()

    @staticmethod
    def _notebook(slots: int) -> tuple[npt.NDArray[np.uint64], IArr, IArr, IArr]:
        return (np.zeros(slots, dtype=np.uint64), np.zeros(slots, dtype=np.int64),
                np.zeros(slots, dtype=np.int64), np.zeros(2, dtype=np.int64))

    def _room(self) -> None:
        """Keep the open-addressing notebook at most half full after the next byte."""
        keys, p, n, meta = self.ad
        if 2 * (int(meta[0]) + 8 * self.n_ad) <= keys.shape[0]:
            return
        bigger = self._notebook(2 * keys.shape[0])
        self.k.rehash(keys, p, n, bigger[0], bigger[1], bigger[2])
        bigger[3][:] = meta
        self.ad = bigger
        self._room()

    def code_byte(self, coder: CoderPort, byte: int) -> int:
        """One template byte through the kernel; stops with a typed error on a broken state."""
        self._room()
        coder.reserve(BYTE_BITS_RESERVE)
        cu, ci, bits = coder.arrays()
        model = (self.hp, self.ip, self.hkind, self.hist, self.ms, self.w, self.xs, self.hb, *self.tables)
        akeys, ap, an, meta = self.ad
        out = int(self.k.cm_byte(1 if coder.encoding else 0, byte, cu, ci, bits, model, self.tabs,
                                 (self.akey, akeys, ap, an, meta)))
        coder.check()
        if int(meta[AD_STATUS]):
            if coder.encoding:
                raise DrawerUnavailable("template_model: weight out of range")
            raise DamagedStream("החבילה פגומה: משקל של מודל התבנית יצא מהטווח")
        return out

    def set_word(self, wid: int) -> None:
        """The word that just filled a slot (distance back to 0)."""
        self.ms[0] = wid
        self.ms[1] = 0


class FlatTemplateModel:
    """The plate: every template bit costs exactly one bit - safe, learns nothing."""

    name = "flat"

    def __init__(self, kernels: KernelSet) -> None:
        self.k = kernels

    def start(self, n: int) -> None:
        """Nothing to forget."""

    def code_byte(self, coder: CoderPort, byte: int) -> int:
        """Eight bits at probability 1/2."""
        coder.reserve(BYTE_BITS_RESERVE)
        cu, ci, bits = coder.arrays()
        out = int(self.k.flat_byte(1 if coder.encoding else 0, byte, cu, ci, bits))
        coder.check()
        return out

    def set_word(self, wid: int) -> None:
        """Words do not matter to a flat model."""
