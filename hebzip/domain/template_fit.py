"""Fitting the template model's start weights on held-out text (the recipe of a ``weights`` table).

Why the real model and the real coder (the change-without-code rule / single source): the shipped start weights are
what the mixer learned after reading held-out texts one after another (the research prototype's
``tp_train.py``: weights carried from text to text, everything else fresh per text). Running the very
``CmTemplateModel`` and ``RangeEncoder`` the codec runs means the fitted weights can never disagree
with the code that reads them; the coded bits are thrown away. Integers only, so the result is the
same on every run and in both engines. Zero I/O.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from hebzip.domain.coder import RangeEncoder
from hebzip.domain.contracts import KernelSet
from hebzip.domain.template_cm import CmTemplateModel

IArr = npt.NDArray[np.int64]
FitText = tuple[bytes, list[int]]
"""One fitting text: its template bytes and the word id of every slot in it."""


def fit_weights(model: CmTemplateModel, texts: list[FitText], passes: int, slot_end: Collection[int],
                kernels: KernelSet) -> IArr:
    """Code every text ``passes`` times, keeping the mixer weights between texts -> the learned weights."""
    for _ in range(passes):
        for template, ids in texts:
            model.carry_weights(len(template))
            coder = RangeEncoder(kernels)
            j = 0
            for byte in template:
                model.code_byte(coder, byte)
                if byte in slot_end:
                    model.set_word(ids[j])
                    j += 1
    return np.asarray(model.weights(), dtype=np.int64)
