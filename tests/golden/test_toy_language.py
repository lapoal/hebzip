"""The change-without-code rule for real: a second language made of config rows + table files only, no code change.

The toy Latin pack (``[languages.latin_toy]`` + modes ``latin_fast`` / ``latin_max`` + ``tables/latin_toy/``,
built from ``langpacks/latin_toy/corpus.txt`` by ``hebzip.ops.build_langpack``) lives in the SAME config as
Hebrew (the language is a row of each mode). Through the real CLI: every sample text
round-trips in separate processes in both model modes and both engines. A package can never be opened as
the other language: the language's rows are part of the format signature, so a Latin package whose
header claims a Hebrew mode, and a Hebrew package read by a config whose ``fast`` names the Latin pack,
are both refused with exit 3. Hebrew golden identity is guarded by ``test_golden_identity.py``.
"""

from __future__ import annotations

import pytest

from tests.support import ENGINES, ROOT, TEXTS, compress_file, config_variant, decompress_file, engine_config, work_dir

SAMPLES = sorted((ROOT / "langpacks" / "latin_toy" / "samples").glob("*.txt"))
TABLE_MISMATCH = 3
MODE_BYTE = 0x80
"""The compact header's first byte = ``0x80 | mode id`` (``domain/header.py``)."""
HEBREW_FAST_ID = 1


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", ["latin_fast", "latin_max"])
def test_toy_language_round_trips_in_separate_processes(mode: str, engine: str) -> None:
    assert len(SAMPLES) >= 3
    config = engine_config(engine, f"toy_cfg_{mode}_{engine}")
    out = work_dir(f"toy_{mode}_{engine}")
    for text in SAMPLES:
        packed, back = out / f"{text.stem}.hz", out / f"{text.stem}.out"
        done = compress_file(text, packed, mode, "--no-fallback", "--config", str(config))
        assert done.returncode == 0, (text.name, done.stderr[-800:])
        assert len(packed.read_bytes()) < len(text.read_bytes()), text.name
        done = decompress_file(packed, back, "--config", str(config))
        assert done.returncode == 0, (text.name, done.stderr[-800:])
        assert back.read_bytes() == text.read_bytes(), text.name


def test_a_package_is_never_opened_as_the_other_language() -> None:
    out = work_dir("toy_cross")
    toy_pkg, he_pkg = out / "toy.hz", out / "he.hz"
    assert compress_file(SAMPLES[0], toy_pkg, "latin_fast", "--no-fallback").returncode == 0
    assert compress_file(TEXTS / "own_story.txt", he_pkg, "fast", "--no-fallback").returncode == 0
    forged = out / "forged.hz"
    data = toy_pkg.read_bytes()
    forged.write_bytes(bytes([MODE_BYTE | HEBREW_FAST_ID]) + data[1:])
    assert decompress_file(forged, out / "a.out").returncode == TABLE_MISMATCH
    latin_fast = config_variant("toy_cross_cfg", ('language = "he"                  # a [languages.*] pack',
                                                  'language = "latin_toy"'))
    assert decompress_file(he_pkg, out / "b.out", "--config", str(latin_fast)).returncode == TABLE_MISMATCH
    assert not (out / "a.out").exists() and not (out / "b.out").exists()
    assert decompress_file(he_pkg, out / "c.out").returncode == 0
    assert (out / "c.out").read_bytes() == (TEXTS / "own_story.txt").read_bytes()
