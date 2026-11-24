"""The range coder's two ends around the shared kernel state (plain code - the change-without-code rule).

Why one object per message and plain arrays inside: the template model codes bits and the word
model codes whole symbols on the SAME interval, so both receive this object and pass its arrays to
their kernels. The encoder keeps one output bit per byte cell (like the research prototype's bit list) and
packs at the end: trailing zero bytes are dropped because the decoder reads zeros past the end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import DamagedStream
from hebzip.domain.contracts import KernelSet
from hebzip.domain.kernels_layout import (
    CI_SIZE,
    FULL_INT,
    NBITS,
    NOUT,
    PEND,
    PMAX,
    QTR_INT,
    REG_BITS,
    STATUS,
    STATUS_TEXT,
)

START_BITS = 1 << 12
"""First size of the encoder's bit buffer (it doubles when needed - memory only, no bit changes)."""


class _CoderState:
    """The state both ends share: ``cu`` = [low, high, value], ``ci`` = the int cells, ``bits``."""

    def __init__(self, bits: npt.NDArray[np.uint8]) -> None:
        self.cu = np.array([0, FULL_INT, 0], dtype=np.uint64)
        self.ci = np.zeros(CI_SIZE, dtype=np.int64)
        self.bits = bits

    def arrays(self) -> tuple[Any, Any, Any]:
        """(cu, ci, bits) for the kernels."""
        return self.cu, self.ci, self.bits


class RangeEncoder(_CoderState):
    """Collects coded bits; ``finish`` returns the body bytes."""

    encoding = True

    def __init__(self, kernels: KernelSet) -> None:
        super().__init__(np.zeros(START_BITS, dtype=np.uint8))
        self.kernels = kernels

    def reserve(self, extra_bits: int) -> None:
        """Grow the bit buffer so ``extra_bits`` more bits (plus every pending bit) fit."""
        need = int(self.ci[NOUT]) + int(self.ci[PEND]) + extra_bits
        if need > self.bits.shape[0]:
            grown = np.zeros(max(need, 2 * self.bits.shape[0]), dtype=np.uint8)
            grown[:self.bits.shape[0]] = self.bits
            self.bits = grown

    def check(self) -> None:
        """Nothing can go wrong while encoding."""

    def finish(self) -> bytes:
        """Flush: one more bit that settles the interval, the pending bits, then pack and strip."""
        self.reserve(REG_BITS)
        bit = 0 if int(self.cu[0]) < QTR_INT else 1
        self.ci[PEND] += 1
        n = int(self.ci[NOUT])
        self.bits[n] = bit
        pending = int(self.ci[PEND])
        self.bits[n + 1:n + 1 + pending] = 1 - bit
        used = n + 1 + pending
        return np.packbits(self.bits[:used]).tobytes().rstrip(b"\x00")


class RangeDecoder(_CoderState):
    """Reads a body; past its end it reads zeros, at most ``phantom_max`` of them."""

    encoding = False

    def __init__(self, kernels: KernelSet, body: bytes, phantom_max: int) -> None:
        super().__init__(np.unpackbits(np.frombuffer(body, dtype=np.uint8)))
        self.kernels = kernels
        self.ci[NBITS] = self.bits.shape[0]
        self.ci[PMAX] = phantom_max
        kernels.dec_start(self.cu, self.ci, self.bits)

    def reserve(self, extra_bits: int) -> None:
        """Decoding writes nothing."""

    def check(self) -> None:
        """Raise ``DamagedStream`` when a kernel flagged the stream (bounded work, typed error)."""
        status = int(self.ci[STATUS])
        if status:
            raise DamagedStream(f"החבילה פגומה: {STATUS_TEXT.get(status, 'מצב מפענח לא מוכר')}")
