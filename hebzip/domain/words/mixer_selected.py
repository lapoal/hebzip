"""Mixer drawer ``trust_selected``: one trust vector per situation, chosen before every word.

Why (the PAQ-family idea "mixer weights selected by context"): the plain ``trust`` mixer learns ONE vector
per text, but which channel deserves trust depends on the situation - after a full stop the gap-start
channels know more, after a word the corpus never saw the text channels do, and the first words of a
message have no text evidence yet. Here the situation key is
    key = (gap class * 2 + [the channel ``select_channel`` found a corpus list]) * 2 + [n >= position_split]
and each key has its own vector, started from the pretrained table ``trust_sets`` (fitted on held-out
corpus; a key missing there starts from the channels' ``trust0``), learnt word by word with the same
Bayes step and share-back as ``trust`` (``mixer.trust_step`` - one copy of that maths).
The plate is ``static`` (unchanged); switching this drawer to ``trust`` gives the old mixer.
"""

from __future__ import annotations

from hebzip.domain.contracts import Look, WordHistory
from hebzip.domain.words.mixer import normalised, trust_step

TrustSets = dict[int, list[float]]


class SelectedTrustMixer:
    """Per-situation trust vectors, learnt word by word."""

    name = "trust_selected"

    def __init__(self, eta: float, alpha: float, sets: TrustSets, select_at: int, position_split: int) -> None:
        self.eta = eta
        self.alpha = alpha
        self.sets = sets
        self.select_at = select_at
        self.position_split = position_split
        self.g0: list[float] = []
        self.live: TrustSets = {}
        self.key = 0

    def start(self, trust0: list[float]) -> None:
        """Every situation back to its pretrained start (or ``trust0``)."""
        self.g0 = list(trust0)
        self.live = {}
        self.key = 0

    def select(self, h: WordHistory, looks: list[Look]) -> None:
        """Pick the situation of the next word."""
        found = 1 if self.select_at >= 0 and looks[self.select_at].found else 0
        late = 1 if h.n >= self.position_split else 0
        self.key = (h.gap.cls * 2 + found) * 2 + late
        if self.key not in self.live:
            start = self.sets.get(self.key)
            self.live[self.key] = list(start) if start else list(self.g0)

    def weights(self, confs: list[float]) -> list[float]:
        """normalise(g_key_i * conf_i)."""
        return normalised(self.live[self.key], confs)

    def update(self, w: list[float], parts: list[float]) -> None:
        """The ``trust`` step on the selected vector only."""
        g = trust_step(self.live[self.key], self.g0, w, parts, self.eta, self.alpha)
        if g is not None:
            self.live[self.key] = g
