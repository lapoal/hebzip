"""Golden identity: every mode reproduces its golden packages byte for byte, in every engine.

Through the real CLI in a child process, every golden text of the mode's language, ``--no-fallback`` so the
mode under test is the one that answers. The WHOLE package is compared - the 2-byte header (``0x80 | mode
id`` + the signature byte, also pinned in ``support.SIGNATURE_BYTE``) and the payload. Both engines
(python and numba run the same kernel source, so they must give the same bytes). Every compress run also
decodes its package in-process (``verify_roundtrip``), so a match is a round trip as well.
Table stores: fast and max run in BOTH stores - ``eager`` (the pinned files) and ``mmap`` (the derived
narrowed copy, made first by the real converter into the test's scratch folder): 2 engines x 2 stores. The
other model modes run in both engines with the store the real config selects; generic needs no tables.
The golden packages are made by ``python -m tests.golden.make_golden`` (never by a test).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hebzip.domain.header import COMPACT_FLAG
from tests.support import (ENGINES, GOLDEN_MODES, HEBREW_TEXTS, SIGNATURE_BYTE, STORES, compress_file,
                           engine_config, golden_package, mode_texts, ready_store_config, real_config, work_dir)

WORKERS = {"max_plus": 2}
"""Parallel CLI runs per mode (max_plus holds the most tables in memory)."""
BOTH_STORES = ("fast", "max")
OTHERS = tuple(m for m in GOLDEN_MODES if m not in BOTH_STORES and m != "generic")


def _one(mode: str, text: Path, out_dir: Path, extra: tuple[str, ...]) -> str | None:
    """A failure description, or ``None`` when this text gives exactly the golden package."""
    out = out_dir / f"{text.stem}.hz"
    done = compress_file(text, out, mode, "--no-fallback", *extra)
    if done.returncode != 0:
        return f"{text.stem}: exit {done.returncode}: {done.stderr[-300:]!r}"
    data, expected = out.read_bytes(), golden_package(mode, text.stem)
    head = bytes((COMPACT_FLAG | real_config()["modes"][mode]["id"], SIGNATURE_BYTE[mode]))
    if data[:2] != head or expected[:2] != head:
        return f"{text.stem}: header {data[:2].hex()} / golden {expected[:2].hex()} is not {head.hex()}"
    if data != expected:
        return f"{text.stem}: package differs from golden ({len(data)} vs {len(expected)} bytes)"
    return None


def _all(mode: str, name: str, extra: tuple[str, ...] = ()) -> list[str]:
    out_dir = work_dir(name)
    texts = mode_texts(mode)
    assert texts and (GOLDEN_MODES[mode] != "he" or len(texts) == HEBREW_TEXTS)
    with ThreadPoolExecutor(WORKERS.get(mode, 4)) as pool:
        results = list(pool.map(lambda t: _one(mode, t, out_dir, extra), texts))
    return [r for r in results if r]


def test_generic_package_equals_golden() -> None:
    failures = _all("generic", "identity_generic")
    assert not failures, failures


def _store_config(mode: str, engine: str, store: str) -> Path:
    """The engine and store rows set; for ``mmap`` the derived copy of the mode's tables is made first."""
    return ready_store_config(f"identity_cfg_{mode}_{engine}_{store}", store, (mode,),
                              ('engine = "numba"', f'engine = "{engine}"'))


@pytest.mark.parametrize("store", STORES)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", BOTH_STORES)
def test_fast_and_max_equal_golden_in_every_engine_and_store(mode: str, engine: str, store: str) -> None:
    cfg = _store_config(mode, engine, store)
    failures = _all(mode, f"identity_{mode}_{engine}_{store}", ("--config", str(cfg)))
    assert not failures, failures


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", OTHERS)
def test_every_other_model_mode_equals_golden_in_every_engine(mode: str, engine: str) -> None:
    cfg = engine_config(engine, f"identity_cfg_{mode}_{engine}")
    failures = _all(mode, f"identity_{mode}_{engine}", ("--config", str(cfg)))
    assert not failures, failures
