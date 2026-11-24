"""Letter-mixer drawers: ``geometric`` (mode max) and the plate ``word_only``.

geometric (the research prototype's option B): p(c) ~ exp(sum_i w_i * log p_i(c)). The weight set is chosen
per node by (min(depth, depth_max), confidence bucket of the word model's best child share
max(m) / sum(m)); a new set starts from the pretrained start sets (a table) when it has one, else
from ``start_weights``; an input that is not mixed starts (and stays) at 0. After each coded letter:
w_i += lr * (log p_i(coded) - E_p[log p_i]).
Exactness: max(m) / sum(m) is Python's correctly rounded int / int, as in the research prototype; every float
sum runs left to right from 0.0 (never
no ``sum()`` of floats); ``exp``/``log`` are the platform's (same machine, same bytes).
word_only (the plate): input 0 alone, weight 1, never learns = the word model by the chain rule.
"""

from __future__ import annotations

from math import exp

WeightSets = dict[tuple[int, int], list[float]]


def softmax_mix(w: list[float], ins: list[list[float] | None]) -> list[float]:
    """normalise(exp(z - max z)), z(c) = sum over mixed inputs in index order of w_i * log p_i(c)."""
    first = ins[0]
    assert first is not None
    z = [0.0] * len(first)
    for i, li in enumerate(ins):
        if li is None:
            continue
        wi = w[i]
        for c in range(len(z)):
            z[c] += wi * li[c]
    top = max(z)
    e = [exp(v - top) for v in z]
    total = 0.0
    for v in e:
        total += v
    return [v / total for v in e]


class GeometricMixer:
    """Weight sets per (depth, confidence), learnt letter by letter."""

    name = "geometric"

    def __init__(self, lr: float, start_weights: list[float], depth_max: int, conf_buckets: list[float],
                 start_sets: WeightSets, mixed: list[bool]) -> None:
        self.lr = lr
        self.on = mixed
        self.w0 = [w if on else 0.0 for w, on in zip(start_weights, mixed, strict=True)]
        self.depth_max = depth_max
        self.buckets = conf_buckets
        self.start_sets = start_sets
        self.start()

    def start(self) -> None:
        """Every set back to its start."""
        self.sets: WeightSets = {}

    def bucket(self, masses: list[int], total: int) -> int:
        """How many thresholds the best child's share reaches."""
        share = max(masses) / total
        cb = 0
        while cb < len(self.buckets) and share >= self.buckets[cb]:
            cb += 1
        return cb

    def select(self, depth: int, masses: list[int], total: int) -> list[float]:
        """The node's weight set (created on first use)."""
        sel = (min(depth, self.depth_max), self.bucket(masses, total))
        w = self.sets.get(sel)
        if w is None:
            pre = self.start_sets.get(sel)
            w = self.sets[sel] = [pre[i] if self.on[i] else 0.0 for i in range(len(self.on))] if pre else list(self.w0)
        return w

    def probs(self, w: list[float], ins: list[list[float] | None]) -> list[float]:
        """The geometric mix."""
        return softmax_mix(w, ins)

    def learn(self, w: list[float], ins: list[list[float] | None], p: list[float], j: int) -> None:
        """One gradient step on the coded child (nothing when lr = 0)."""
        if self.lr > 0.0:
            for i, li in enumerate(ins):
                if li is None:
                    continue
                ex = 0.0
                for c in range(len(p)):
                    ex += p[c] * li[c]
                w[i] += self.lr * (li[j] - ex)


class WordOnlyMixer:
    """The plate: the word model alone (weight 1), never learns."""

    name = "word_only"

    def __init__(self, inputs: int) -> None:
        self.w = [1.0] + [0.0] * inputs

    def start(self) -> None:
        """Nothing learnt."""

    def select(self, depth: int, masses: list[int], total: int) -> list[float]:
        """Always (1, 0, 0, ...)."""
        return self.w

    def probs(self, w: list[float], ins: list[list[float] | None]) -> list[float]:
        """The word model's own letter shares."""
        return softmax_mix(w, [ins[0]] + [None] * (len(ins) - 1))

    def learn(self, w: list[float], ins: list[list[float] | None], p: list[float], j: int) -> None:
        """Never learns."""
