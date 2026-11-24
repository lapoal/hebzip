"""The word coder ``spelled`` (mode max): every word id is spelled over the alphabetical tree.

Why (the letter-model study, option B): the word table already gives every word an exact integer
weight; summed over a tree node's children they are the word model's view of the next LETTER. Mixing
that view with letter models (static corpus chains + the text's own letters) predicts unseen and
rare words far better than one symbol over 1.2M ids.
Per word (the research prototype's ``ls_model.LModel.code_word``, step by step):
  1. masses of every rank range: a * (dense base cumsum in alphabetical order) + the sparse parts'
     weights of the words in each child (integer sums - the same as F(x) of the direct coder; the
     weights are summed per child of each node and narrowed to the coded child, instead of being
     sorted by rank once per word - integer sums in any order, so the same masses)
  2. walk from the root: a node with one child costs nothing; otherwise input 0 = log(m_c / sum m),
     the letter inputs give their logs, the letter mixer mixes, the child is coded on the range
     coder with freq = int(p * 2**q_bits) + 1 (every child stays codable), and the mixer learns
  3. the walk ends on a range of one word or on the word-end child; every letter (and the end) of
     the word is then observed by the letter inputs, and the history moves on.
Damage: the decoder's budget is checked after every coded letter (design decision).
"""

from __future__ import annotations

from math import log

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import DamagedStream, DrawerUnavailable
from hebzip.domain.contracts import BasePriorPort, CoderPort, CodingTablePort, LetterInputPort, LetterMixerPort
from hebzip.domain.mixtable import MAX_TOTAL
from hebzip.domain.spell.trie import END, AlphaIndex

IArr = npt.NDArray[np.int64]
LETTER_BITS_RESERVE = 96
"""Room in the encoder's bit buffer for one coded letter (a letter costs at most ~q_bits + pending)."""
NODE_MEMO_LIMIT = 200_000
"""Cleared when larger (memory only): the tree and the base are fixed, so a node's children and their
dense masses are the same every time it is visited (speed only; a pure cache, no bit depends on it)."""
Node = tuple[list[int], list[int], IArr, IArr]
"""A visited node: child codes, child bounds (ranks), the bounds as an array, the dense mass per child."""


class SpelledWordCoder:
    """Letter-by-letter coding of a word id over the alphabetical tree."""

    name = "spelled"

    def __init__(self, index: AlphaIndex, inputs: list[LetterInputPort], mixer: LetterMixerPort, q_bits: int,
                 history: int) -> None:
        self.index = index
        self.inputs = inputs
        self.mixer = mixer
        self.q_one = 1 << q_bits
        self.history = history
        self._base: BasePriorPort | None = None
        self._dense = np.zeros(1, dtype=np.int64)
        self._nodes: dict[tuple[int, int, int], Node] = {}
        self.start()

    def start(self) -> None:
        """A new message: empty letter history (``history`` symbols of "nothing"), inputs and mixer reset."""
        self.hist = [-1] * self.history
        for inp in self.inputs:
            inp.start()
        self.mixer.start()

    def _dense_cum(self, base: BasePriorPort) -> IArr:
        """[0, U[perm[0]], ...] cumulative in alphabetical order (computed once per base)."""
        if self._base is not base:
            u = np.diff(base.cum)
            self._dense = np.concatenate([[0], np.cumsum(u[self.index.perm])]).astype(np.int64)
            self._base = base
            self._nodes = {}
        return self._dense

    def _node(self, dense: IArr, lo: int, hi: int, depth: int) -> Node:
        """The children of [lo, hi) at ``depth`` with their dense masses (cached: tree and base are fixed)."""
        key = (lo, hi, depth)
        node = self._nodes.get(key)
        if node is None:
            codes, bounds = self.index.children(lo, hi, depth)
            bnd = np.array(bounds, dtype=np.int64)
            node = (codes, bounds, bnd, dense[bnd[1:]] - dense[bnd[:-1]])
            if len(self._nodes) > NODE_MEMO_LIMIT:
                self._nodes.clear()
            self._nodes[key] = node
        return node

    def _sparse(self, table: CodingTablePort) -> tuple[IArr, IArr]:
        """Every sparse weight with the rank of its word: (ranks, weights), in no particular order."""
        if not table.parts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        ids = np.concatenate([p.ids for p in table.parts])
        wts = np.concatenate([np.diff(p.cum) for p in table.parts])
        return self.index.rank_of[ids], wts

    def code(self, coder: CoderPort, table: CodingTablePort, x: int) -> int:
        """Encode ``x`` / decode one word id, letter by letter."""
        if table.total > MAX_TOTAL:
            if coder.encoding:
                raise DrawerUnavailable("word_model: table total above 2**62")
            raise DamagedStream("החבילה פגומה: טבלת מילים גדולה מדי")
        dense, (ranks, wts) = self._dense_cum(table.base), self._sparse(table)
        target = self.index.codes(x) + [END] if coder.encoding else None
        lo, hi, depth = 0, self.index.n, 0
        pre: list[int] = []
        while hi - lo > 1:
            codes, bounds, bnd, dense_m = self._node(dense, lo, hi, depth)
            j = 0
            if len(codes) > 1:
                child = np.searchsorted(bnd, ranks, side="right") - 1
                sparse_m = np.zeros(len(codes), dtype=np.int64)
                np.add.at(sparse_m, child, wts)
                masses = (table.a * dense_m + sparse_m).tolist()
                want = codes.index(target[depth]) if target is not None else 0
                j = self._letter(coder, codes, masses, self.hist + pre, depth, want)
                inside = child == j
                ranks, wts = ranks[inside], wts[inside]
            c = codes[j]
            lo, hi = bounds[j], bounds[j + 1]
            pre.append(c)
            depth += 1
            if c == END:
                break
        wid = self.index.word_at(lo)
        if coder.encoding and wid != x:
            raise DrawerUnavailable("word_coder: the spelled walk did not reach the word")
        self._commit(self.index.codes(wid))
        return wid

    def _letter(self, coder: CoderPort, codes: list[int], masses: list[int], hist: list[int], depth: int,
                want: int) -> int:
        """Mix and code one node's child; the mixer learns it."""
        total = 0
        for v in masses:
            total += v
        log_total = log(total)
        ins: list[list[float] | None] = [[log(v) - log_total for v in masses]]
        ins += [inp.logs(hist, codes) if inp.active else None for inp in self.inputs]
        w = self.mixer.select(depth, masses, total)
        p = self.mixer.probs(w, ins)
        j = self._code_child(coder, [int(v * self.q_one) + 1 for v in p], want)
        self.mixer.learn(w, ins, p, j)
        return j

    def _code_child(self, coder: CoderPort, freq: list[int], want: int) -> int:
        cum = [0]
        for f in freq:
            cum.append(cum[-1] + f)
        total = cum[-1]
        k = coder.kernels
        coder.reserve(LETTER_BITS_RESERVE)
        cu, ci, bits = coder.arrays()
        if coder.encoding:
            k.enc_symbol(cu, ci, bits, cum[want], cum[want + 1], total)
            return want
        target = int(k.dec_target(cu, total))  # < total: the state was checked after the last step
        j = 0
        while j + 2 < len(cum) and cum[j + 1] <= target:
            j += 1
        k.dec_symbol(cu, ci, bits, cum[j], cum[j + 1], total)
        coder.check()
        return j

    def _commit(self, letters: list[int]) -> None:
        """The word is decided: every input observes its letters and the end; the history moves on."""
        h = self.hist
        for c in [*letters, END]:
            for inp in self.inputs:
                if inp.active:
                    inp.observe(h, c)
            h = h[1:] + [c]
        self.hist = h
