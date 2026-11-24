"""Drawers that read a table straight from the store give the same numbers in every integer width.

Why: the mapped store hands the corpus word pairs and the letter chains out as they lie on disk - the
pinned width (the Hebrew pairs are uint32, the chain counts int32) or a narrowed one - so a mapped file is
not copied into int64 at start-up. Those drawers must therefore never let the width reach their numbers.
The trap found here: ``np.cumsum`` of a uint32 slice is uint64, and a uint64 id list mixed with int64 in the
coding table becomes float64. Each drawer is built from the int64 arrays (the old
loader's copy), the pinned arrays and the narrowed arrays, and every answer must be identical - the ids
of a sparse part also in dtype int64.
"""

from __future__ import annotations

import random
import tracemalloc
from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.config.load import load_config
from hebzip.domain.contracts import WordHistory
from hebzip.domain.spell.chains import FAMILY, WITHIN, LetterChain
from hebzip.domain.words.predictors import CorpusPairs
from hebzip.services.assemble import PAIR_ARRAYS
from hebzip.services.select import REAL
from hebzip.store.narrow import narrow
from tests.support import CONFIG, ROOT

ALL_BITS = [8, 16, 32, 64]
SETTINGS = load_config(CONFIG)
STORE = REAL["table_store"]["eager"](SETTINGS, ROOT)
Arrays = dict[str, npt.NDArray[Any]]


def _widths(arrays: Arrays) -> list[Arrays]:
    """The same arrays as int64 (the old copy), as pinned, and narrowed."""
    return [{k: v.astype(np.int64) for k, v in arrays.items()}, dict(arrays),
            {k: narrow(v, ALL_BITS) for k, v in arrays.items()}]


def _answers(pairs: CorpusPairs, prevs: list[int | None]) -> tuple[list[object], set[str]]:
    """Every answer of the drawer, and the dtypes of the sparse id lists it built."""
    out: list[object] = []
    dtypes: set[str] = set()
    for prev in prevs:
        look = pairs.look(WordHistory(prev=prev))
        part = pairs.sparse(0.37, 1 << 40)
        if part is not None:
            dtypes.add(part.ids.dtype.str)
            out.append((part.ids.tolist(), part.cum.tolist()))
        out.append((look, [pairs.prob(x) for x in (0, 1, 7, 1000, prev or 0)]))
    return out, dtypes


def test_corpus_pairs_answer_the_same_in_every_width() -> None:
    raw = STORE.arrays("bigram_mc1")
    rng = random.Random(20)
    ctx = raw["ctx"]
    prevs: list[int | None] = [None, 0, 1, 2] + [int(ctx[rng.randrange(len(ctx))]) for _ in range(300)]
    answers, dtypes = [], set()
    for arrays in _widths({k: raw[k] for k in PAIR_ARRAYS}):
        ctx, ctx_n, off, nxt_d, cnt = (arrays[k] for k in PAIR_ARRAYS)
        found, kinds = _answers(CorpusPairs(ctx, ctx_n, off, nxt_d, cnt, 0.85, 1.0), prevs)
        answers.append(found)
        dtypes |= kinds
    assert answers[0] == answers[1] == answers[2]
    assert dtypes == {"<i8"}, dtypes


def _chain(arrays: Arrays, context: str) -> LetterChain:
    row = SETTINGS.modes["max"].letter_chains[context]
    family = FAMILY[context]
    entries = [(arrays[f"{family}{k}_ck"], arrays[f"{family}{k}_cn"]) for k in range(1, row.max_order + 1)]
    return LetterChain(arrays["n0"].astype(np.int64), entries, WITHIN[context], row.beta, row.add_count,
                       row.symbol_bits)


def test_letter_chains_answer_the_same_in_every_width() -> None:
    raw = STORE.arrays("letters_v15")
    rng = random.Random(15)
    symbols = len(raw["n0"])
    hists = [[rng.randrange(symbols) for _ in range(rng.randrange(0, 9))] for _ in range(400)]
    for context in ("cross", "within"):
        answers = [[_chain(arrays, context).probs(h) for h in hists] for arrays in _widths(dict(raw))]
        assert answers[0] == answers[1] == answers[2], context


def test_a_narrowed_chain_is_searched_without_copying_it() -> None:
    """A Python-int search key made numpy cast the whole narrowed key array on every lookup (max 4x
    slower on the mapped copy, same bytes). Measured by allocation, not time: 300 lookups in a 2M-key int32
    order must allocate far less than one int64 copy of it (16 MB)."""
    keys = np.arange(0, 4_000_000, 2, dtype=np.int32) << 2
    counts = np.ones(len(keys), dtype=np.int16)
    chain = LetterChain(np.ones(3, dtype=np.int64), [(keys, counts)], False, 1.0, 1.0, 2)
    tracemalloc.start()
    for key in range(0, 3_000_000, 10_000):
        chain.counts(1, key)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 1 << 20, f"{peak} bytes - the key array was copied"
    assert chain.counts(1, 1 << 40) is None  # above every key of the type: no hit, no overflow
    wide = LetterChain(np.ones(3, dtype=np.int64), [(keys.astype(np.int64), counts)], False, 1.0, 1.0, 2)
    for key in (0, 1, 2, 99_999, 2_000_000, 3_999_998, 1 << 29, (1 << 31) - 1, 1 << 31, 1 << 40):
        assert chain.counts(1, key) == wide.counts(1, key), key
