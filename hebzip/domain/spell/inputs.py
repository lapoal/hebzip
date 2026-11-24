"""Letter-input drawers of the speller: ``static_chain``, ``text_letters`` and the plate ``null``.

``static_chain``  a corpus letter chain (``chains.py``): the cross-word chain (inputs "x" of the
                  research prototype) or the within-word chain ("w"), named by the row ``chain``.
``text_letters``  this text's own letter counts over cross-word contexts, orders 1..``max_order``,
                  backed off (weight ``gamma``) to a static chain named by ``base_chain`` - the
                  research prototype's input "t", which leaned on the cross-word chain. The base chain is a
                  table + rows of its own (``[letter_chains.*]``), so switching the cross-word input
                  off leaves text_letters exactly as it was.
``null``          the plate: inactive - the mixer leaves it out (the research prototype's ``ins[i] = None``).
"""

from __future__ import annotations

from math import log

from hebzip.domain.spell.chains import LetterChain, keys_from_hist


class StaticChainInput:
    """log P from one static chain."""

    name = "static_chain"
    active = True

    def __init__(self, chain: LetterChain) -> None:
        self.chain = chain
        self.reach = chain.orders

    def start(self) -> None:
        """Static knowledge - nothing to forget."""

    def logs(self, hist: list[int], codes: list[int]) -> list[float]:
        """log P(c) for each child code."""
        lp = self.chain.logs(hist)
        return [lp[c] for c in codes]

    def observe(self, hist: list[int], symbol: int) -> None:
        """The corpus does not learn."""


class TextLetters:
    """log P from the text's letter counts, backed off to a static chain."""

    name = "text_letters"
    active = True

    def __init__(self, base: LetterChain, max_order: int, gamma: float) -> None:
        self.base = base
        self.orders = max_order
        self.gamma = gamma
        self.symbols = base.symbols
        self.reach = max(max_order, base.orders)
        self.start()

    def start(self) -> None:
        """Empty notebook: (order, context key) -> counts per symbol + their total (last cell)."""
        self.counts: dict[tuple[int, int], list[int]] = {}

    def logs(self, hist: list[int], codes: list[int]) -> list[float]:
        """p_k(c) = (gamma * p_{k-1}(c) + n_k(c)) / (N_k + gamma) while the text has the context - the same
        float expressions in the same order as the one-symbol-at-a-time form, per child symbol."""
        p = self.base.probs(hist)
        g = self.gamma
        keys = keys_from_hist(hist, self.orders, False, self.base.base)
        seen: list[list[int]] = []
        for k in range(1, self.orders + 1):
            e = self.counts.get((k, keys[k - 1]))
            if e is None:
                break
            seen.append(e)
        out = []
        for c in codes:  # each symbol's chain is independent - only the node's children are computed
            v = p[c]
            for e in seen:
                den = e[self.symbols] + g
                v = g * v / den
                if e[c]:
                    v += e[c] / den
            out.append(log(v))
        return out

    def observe(self, hist: list[int], symbol: int) -> None:
        """Count ``symbol`` after every order of the context ``hist``."""
        keys = keys_from_hist(hist, self.orders, False, self.base.base)
        for k in range(1, self.orders + 1):
            e = self.counts.get((k, keys[k - 1]))
            if e is None:
                e = self.counts[(k, keys[k - 1])] = [0] * (self.symbols + 1)
            e[symbol] += 1
            e[self.symbols] += 1


class NullLetterInput:
    """The plate: not mixed, learns nothing."""

    name = "null"
    active = False
    reach = 0

    def start(self) -> None:
        """Nothing."""

    def logs(self, hist: list[int], codes: list[int]) -> list[float]:
        """Never asked (inactive); no knowledge."""
        return []

    def observe(self, hist: list[int], symbol: int) -> None:
        """Nothing."""
