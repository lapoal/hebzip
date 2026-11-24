"""Round trip across processes (design decision): compress in one interpreter, decompress in ANOTHER.

The decompressor process sees only the package bytes and the config (the one-process-per-test-file rule). All 13
Hebrew golden texts, compared byte for byte. The default config falls back to ``generic`` with
``smaller`` (the fallback path itself is proved in ``services/test_fallback.py``); ``--no-fallback``
runs prove fast, max and max_plus end to end on their own. One more run has the checksum switched on
(extended header), and fast and max run once more in the python engine.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.support import compress_file, config_variant, decompress_file, engine_config, text_paths, work_dir

WORKERS = 3
NO_FALLBACK = "--no-fallback"


def _round(mode: str, text: Path, out_dir: Path, extra: tuple[str, ...] = ()) -> str | None:
    pkg = out_dir / f"{text.stem}.hz"
    back = out_dir / f"{text.stem}.out"
    first = compress_file(text, pkg, mode, *extra)
    if first.returncode != 0:
        return f"{text.stem}: compress exit {first.returncode}: {first.stderr[-300:]!r}"
    second = decompress_file(pkg, back, *(a for a in extra if a != NO_FALLBACK))  # a compress-only flag
    if second.returncode != 0:
        return f"{text.stem}: decompress exit {second.returncode}: {second.stderr[-300:]!r}"
    return None if back.read_bytes() == text.read_bytes() else f"{text.stem}: text differs after round trip"


def _all(mode: str, name: str, extra: tuple[str, ...] = ()) -> list[str]:
    out_dir = work_dir(name)
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda t: _round(mode, t, out_dir, extra), text_paths()))
    return [r for r in results if r]


@pytest.mark.parametrize("mode", ["fast", "max", "generic"])
def test_round_trip_in_separate_processes(mode: str) -> None:
    failures = _all(mode, f"roundtrip_{mode}")
    assert not failures, failures


@pytest.mark.parametrize("mode", ["fast", "max", "max_plus"])
def test_round_trip_of_the_model_alone(mode: str) -> None:
    failures = _all(mode, f"roundtrip_{mode}_alone", (NO_FALLBACK,))
    assert not failures, failures


def test_round_trip_with_checksum_on() -> None:
    cfg = config_variant("roundtrip_crc", ('integrity = "none"', 'integrity = "crc16"'))
    failures = _all("fast", "roundtrip_crc_run", ("--config", str(cfg)))
    assert not failures, failures


@pytest.mark.parametrize("mode", ["fast", "max"])
def test_round_trip_python_engine(mode: str) -> None:
    cfg = engine_config("python", f"roundtrip_python_{mode}")
    failures = _all(mode, f"roundtrip_python_run_{mode}", ("--config", str(cfg)))
    assert not failures, failures
