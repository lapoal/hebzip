"""Base-prior drawers: ``unigram_zipf`` (corpus unigram blended with Zipf) and the plate ``uniform``.

Why a dense base under every word: one range-coding step per word, and every word keeps a weight
a*U[x] >= 1 - nothing is ever uncodable, whatever the predictors say.
U[x] = floor(p[x] * 2**u_scale_bits) + 1 with p = (1-zeta) * unigram share + zeta * Zipf share,
Zipf weight = 2**zipf_weight_bits // (id + zipf_offset) (ids are frequency ranks).
Exactness: the sums are of integer-valued float64s far below 2**53, so they are exact in any order;
the rest are element-wise IEEE operations - the same U on every machine (and in both engines).
The plate gives every id weight 1: no knowledge, still every word codable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hebzip.domain.words.notebook import cumulative as _cumulative

IArr = npt.NDArray[np.int64]


class UnigramZipfBase:
    """The corpus unigram + Zipf base of mode fast."""

    name = "unigram_zipf"

    def __init__(self, unigram: npt.NDArray[np.generic], zeta: float, u_scale_bits: int,
                 zipf_weight_bits: int, zipf_offset: int) -> None:
        uni = unigram.astype(np.float64)
        n = len(uni)
        zipf = ((1 << zipf_weight_bits) // (np.arange(n, dtype=np.int64) + zipf_offset)).astype(np.float64)
        p = (uni * ((1.0 - zeta) / float(uni.sum()))) + zipf * (zeta / float(zipf.sum()))
        self.u = np.floor(p * float(1 << u_scale_bits)).astype(np.int64) + 1
        self.cum = _cumulative(self.u)
        self.total = int(self.cum[-1])

    def prob(self, x: int) -> float:
        """B(x) = U[x] / total (int / int is correctly rounded)."""
        return int(self.u[x]) / self.total


class UniformBase:
    """The plate: every word id weighs 1."""

    name = "uniform"

    def __init__(self, n: int) -> None:
        self.u = np.ones(n, dtype=np.int64)
        self.cum = _cumulative(self.u)
        self.total = n

    def prob(self, x: int) -> float:
        """1 / n."""
        return 1 / self.total
