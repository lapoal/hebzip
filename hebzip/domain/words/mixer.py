"""Mixer drawers: ``trust`` (learns each channel's trust per text) and the plate ``static``.

trust: w_i = g_i * conf_i / sum (sum over channels in config order, left to right from 0.0 - the
research prototype's raw[0] + raw[1] + raw[2] gives the same floats). After each word every channel that
had weight is rescaled by how well it predicted (linear Bayes step, rate ``eta``), then all are
normalised and pulled back toward their start trust (share ``alpha``).
static (the plate): the same weights from the start trust, never learns - safe and simple.
When no channel has evidence the weights are all 0.0; the mixture then gives the base everything.
"""

from __future__ import annotations

from hebzip.domain.contracts import Look, WordHistory


def _total(values: list[float]) -> float:
    """Left-to-right float sum from 0.0 (never ``sum()``: CPython 3.12 compensates float sums)."""
    s = 0.0
    for v in values:
        s += v
    return s


def normalised(g: list[float], confs: list[float]) -> list[float]:
    """normalise(g_i * conf_i); all 0.0 when no channel has evidence."""
    raw = [gi * ci for gi, ci in zip(g, confs, strict=True)]
    s = _total(raw)
    return [r / s for r in raw] if s > 0.0 else [0.0] * len(raw)


def trust_step(g_now: list[float], g0: list[float], w: list[float], parts: list[float], eta: float,
               alpha: float) -> list[float] | None:
    """g_i *= (1 - eta + eta * p_i / mix) where w_i > 0; normalise; share alpha back to g0.

    ``None`` when mix == 0: no channel gave the word any probability (it was coded from the base floor
    alone - possible only when no channel backs off to the base, e.g. corpus_pairs switched off), so
    there is nothing to learn and the trust stays. One copy of this
    step for every trust mixer (``trust_selected`` applies it to the selected vector)."""
    mix = _total([wi * pi for wi, pi in zip(w, parts, strict=True)])
    if mix <= 0.0:
        return None
    g = list(g_now)
    for i, wi in enumerate(w):
        if wi > 0.0:
            g[i] = g[i] * (1.0 - eta + eta * parts[i] / mix)
    s = _total(g)
    s0 = _total(g0)
    return [(1.0 - alpha) * g[i] / s + alpha * g0[i] / s0 for i in range(len(g))]


class TrustMixer:
    """Per-text trust, learnt word by word."""

    name = "trust"

    def __init__(self, eta: float, alpha: float) -> None:
        self.eta = eta
        self.alpha = alpha
        self.g0: list[float] = []
        self.g: list[float] = []

    def start(self, trust0: list[float]) -> None:
        """Trust back to the shipped start values."""
        self.g0 = list(trust0)
        self.g = list(trust0)

    def select(self, h: WordHistory, looks: list[Look]) -> None:
        """One vector for every situation - nothing to pick."""

    def weights(self, confs: list[float]) -> list[float]:
        """normalise(g_i * conf_i)."""
        return normalised(self.g, confs)

    def update(self, w: list[float], parts: list[float]) -> None:
        """The trust step (``trust_step``); no change when mix == 0."""
        g = trust_step(self.g, self.g0, w, parts, self.eta, self.alpha)
        if g is not None:
            self.g = g


class StaticMixer:
    """The plate: start trust only, no learning."""

    name = "static"

    def __init__(self) -> None:
        self.g: list[float] = []

    def start(self, trust0: list[float]) -> None:
        """Keep the start trust."""
        self.g = list(trust0)

    def select(self, h: WordHistory, looks: list[Look]) -> None:
        """Nothing to pick."""

    def weights(self, confs: list[float]) -> list[float]:
        """normalise(g0_i * conf_i)."""
        return normalised(self.g, confs)

    def update(self, w: list[float], parts: list[float]) -> None:
        """Never learns."""
