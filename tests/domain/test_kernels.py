"""The kernels are exact: checked in plain Python against Python's unbounded integers.

The same source is compiled in the numba engine; that it gives the same bytes is checked end to end
by the golden identity test in both engines. Here: the 128-bit helpers on random and edge values,
the word table against a brute-force F, and the symbol coder on random tables.
"""

from __future__ import annotations

import random

import numpy as np

from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.services.select import engine_kernels
from tests.support import ROOT

K = engine_kernels("python", ROOT)
U = np.uint64
EDGES = [0, 1, 2, 2**32 - 1, 2**32, 2**62, 2**63 - 1, 2**63, 2**64 - 2, 2**64 - 1]


def _arith() -> object:
    import importlib
    return importlib.import_module("hebzip.domain.kernels.arith")


def test_wide_arithmetic_matches_python_integers() -> None:
    a_mod = _arith()
    rnd = random.Random(5)
    for _ in range(2000):
        a = rnd.choice([*EDGES, rnd.getrandbits(64)])
        b = rnd.choice([*EDGES, rnd.getrandbits(64)])
        hi, lo = a_mod.mul_wide(U(a), U(b))  # type: ignore[attr-defined]
        assert (int(hi) << 64 | int(lo)) == a * b
        t = rnd.choice([1, 2, 3, 2**62, rnd.getrandbits(62) + 1, 2**63])
        x = rnd.randrange(t)
        assert int(a_mod.scale_down(U(a), U(x), U(t))) == (a + 1) * x // t  # type: ignore[attr-defined]
        d = rnd.choice([1, 2**63, 2**64 - 1, rnd.getrandbits(64) | 1])
        h, low = (rnd.randrange(d) if d > 1 else 0), rnd.getrandbits(64)
        assert int(a_mod.div_wide(U(h), U(low), U(d))) == ((h << 64) | low) // d  # type: ignore[attr-defined]


def _table(rnd: random.Random) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = rnd.randrange(5, 300)
    cum_u = np.concatenate([[0], np.cumsum(np.array([rnd.randrange(1, 9) for _ in range(n)]))]).astype(np.int64)
    ids_parts, cum_parts, offs = [], [], [0]
    for _ in range(rnd.randrange(0, 4)):
        ids = np.array(sorted(rnd.sample(range(n), rnd.randrange(0, min(n, 20)))), dtype=np.int64)
        w = np.array([rnd.randrange(0, 2**40) for _ in ids], dtype=np.int64)
        ids_parts.append(ids)
        cum_parts.append(np.concatenate([[0], np.cumsum(w)]).astype(np.int64))
        offs.append(offs[-1] + len(ids))
    ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.int64)
    cums = np.concatenate(cum_parts) if cum_parts else np.zeros(0, np.int64)
    return rnd.randrange(1, 2**20), cum_u, ids.astype(np.int64), cums, np.array(offs, dtype=np.int64)


def _brute(x: int, a: int, cum_u: np.ndarray, ids: np.ndarray, cums: np.ndarray, offs: np.ndarray) -> int:
    v = a * int(cum_u[x])
    for k in range(len(offs) - 1):
        part = ids[offs[k]:offs[k + 1]]
        v += int(cums[offs[k] + k + int(np.searchsorted(part, x))])
    return v


def test_word_table_and_symbol_coder_are_exact() -> None:
    rnd = random.Random(11)
    for _ in range(40):
        a, cum_u, ids, cums, offs = _table(rnd)
        n = len(cum_u) - 1
        total = _brute(n, a, cum_u, ids, cums, offs)
        for x in rnd.sample(range(n + 1), min(n + 1, 12)):
            assert int(K.table_value(x, a, cum_u, ids, cums, offs)) == _brute(x, a, cum_u, ids, cums, offs)
        sent = [rnd.randrange(n) for _ in range(30)]
        enc = RangeEncoder(K)
        for x in sent:
            enc.reserve(256)
            cu, ci, bits = enc.arrays()
            K.enc_symbol(cu, ci, bits, _brute(x, a, cum_u, ids, cums, offs),
                         _brute(x + 1, a, cum_u, ids, cums, offs), total)
        dec = RangeDecoder(K, enc.finish(), 1024)
        cu, ci, bits = dec.arrays()
        for x in sent:
            target = int(K.dec_target(cu, total))
            got = int(K.table_find(target, a, cum_u, ids, cums, offs))
            K.dec_symbol(cu, ci, bits, _brute(got, a, cum_u, ids, cums, offs),
                         _brute(got + 1, a, cum_u, ids, cums, offs), total)
            assert got == x
        dec.check()
