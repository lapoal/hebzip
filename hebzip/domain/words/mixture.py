"""The word model skeleton ``mixture`` (plain code) and the plate ``null``.

One word = one range-coding step over a table built from drawers:
  1. every ACTIVE predictor looks at the history -> (evidence, escape); plates are left out
  2. the mixer picks its situation (``select``: history + looks; only ``trust_selected`` uses it) and
     turns evidence into weights w_i
  3. dense part a = max(1, int(sum_i w_i * escape_i * scale / base.total))  (all to the base when no
     channel has evidence); each predictor adds its sparse part at its weight
  4. the word coder drawer codes the word id: ``direct`` = one step through F(x) (``kernels/table.py``),
     ``spelled`` = letter by letter (``domain/spell``)
  5. channel probability of the coded word = own prob + escape * base prob -> mixer learns;
     predictors observe; the history advances.
Why this order and these sums: they are the research prototype's operations in the research prototype's order, so
the same floats come out; a new predictor is a new drawer + rows, never a new branch.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import DrawerUnavailable
from hebzip.domain.contracts import (
    BasePriorPort,
    CoderPort,
    Gap,
    KernelSet,
    MixerPort,
    PredictorPort,
    WordCoderPort,
    WordHistory,
)
from hebzip.domain.mixtable import WordTable

IArr = npt.NDArray[np.int64]
WORD_BITS_RESERVE = 192


class MixtureWordModel:
    """Base prior + predictor channels + mixer -> one coded word id."""

    name = "mixture"

    def __init__(self, base: BasePriorPort, predictors: list[PredictorPort], mixer: MixerPort,
                 scale_bits: int, kernels: KernelSet, word_coder: WordCoderPort) -> None:
        self.base = base
        self.channels = [p for p in predictors if p.active]
        self.mixer = mixer
        self.word_coder = word_coder
        self.scale = 1 << scale_bits
        self.k = kernels
        self.start()

    def start(self) -> None:
        """A new message: empty history, every channel and the mixer back to their start."""
        self.h = WordHistory()
        for p in self.channels:
            p.start()
        self.mixer.start([p.trust0 for p in self.channels])
        self.word_coder.start()

    def _table(self, w: list[float], escapes: list[float]) -> WordTable:
        mass = 0.0
        for wi, esc in zip(w, escapes, strict=True):
            mass += wi * esc
        if not any(wi > 0.0 for wi in w):
            mass = 1.0
        a = max(1, int(mass * self.scale / self.base.total))
        parts = [p.sparse(wi, self.scale) for p, wi in zip(self.channels, w, strict=True)]
        return WordTable(a, self.base, [s for s in parts if s is not None], self.k)

    def code(self, coder: CoderPort, x: int, gap: Gap) -> int:
        """Encode ``x`` / decode a word id; then everybody learns it."""
        self.h.gap = gap
        looks = [p.look(self.h) for p in self.channels]
        self.mixer.select(self.h, looks)
        w = self.mixer.weights([lk.conf for lk in looks])
        table = self._table(w, [lk.escape for lk in looks])
        coder.reserve(WORD_BITS_RESERVE)
        x = self.word_coder.code(coder, table, x)
        coder.check()
        b = self.base.prob(x)
        self.mixer.update(w, [p.prob(x) + lk.escape * b for p, lk in zip(self.channels, looks, strict=True)])
        for p in self.channels:
            p.observe(x, self.h)
        self.h.advance(x)
        return x


class NullWordModel:
    """The plate: refuses - a slot without a word model must not decode to a guessed word."""

    name = "null"

    def start(self) -> None:
        """Nothing."""

    def code(self, coder: CoderPort, x: int, gap: Gap) -> int:
        """Refuses (the compressor then falls back to the generic codec)."""
        raise DrawerUnavailable("word_model")
