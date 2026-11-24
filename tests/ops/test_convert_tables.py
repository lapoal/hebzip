"""The table converter and the mapped store through the real paths.

Why child processes (the real-path rule and the one-process-per-test-file rule): the user runs ``python -m
hebzip.ops.convert_tables`` and then
``python -m hebzip compress`` with ``[hebzip].table_store = "mmap"``; these tests do exactly that, into a
scratch copy (``tests/_work/<name>/mm``) - the real ``tables/mmap`` is never touched. Proven here:
  * the mapped store hands out the same values as the pinned files, in widths no wider than the source;
  * a value that fits no allowed width stops the converter with exit 3 and NOTHING is written;
  * a damaged, edited, foreign or incomplete derived copy is refused at start-up (exit 3, no output);
  * both stores give byte-identical packages that round-trip exactly (the toy Latin pack here; Hebrew
    fast and max on all golden texts in ``tests/golden/test_golden_identity.py``).
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from hebzip.config.load import load_config
from hebzip.config.tables import mode_tables
from hebzip.services.select import REAL
from hebzip.store.manifest import MANIFEST, seal
from hebzip.store.port import TableStorePort
from tests.support import ROOT, WORK, compress_file, convert_tables, decompress_file, ready_store_config, store_config

LATIN = ("latin_fast", "latin_max")
SAMPLES = sorted((ROOT / "langpacks" / "latin_toy" / "samples").glob("*.txt"))


def _converted(name: str, *replacements: tuple[str, str]) -> tuple[Path, Path]:
    return ready_store_config(name, "mmap", LATIN, *replacements), WORK / name / "mm"


def _store(cfg: Path, word: str) -> TableStorePort:
    store: TableStorePort = REAL["table_store"][word](load_config(cfg), ROOT)
    return store


def _compare_arrays(name: str, eager: TableStorePort, mapped: TableStorePort, npz: bool) -> int:
    a = eager.arrays(name) if npz else {"": eager.array(name)}
    b = mapped.arrays(name) if npz else {"": mapped.array(name)}
    assert sorted(a) == sorted(b), name
    for key, wide in a.items():
        got = b[key]
        assert not got.flags.owndata and not got.flags.writeable, f"{name}/{key} is not a read-only map"
        assert got.shape == wide.shape and np.array_equal(got, wide), f"{name}/{key}"
        assert got.dtype.kind == wide.dtype.kind and got.dtype.itemsize <= wide.dtype.itemsize, f"{name}/{key}"
    return len(a)


def _compare(cfg: Path) -> int:
    """Every table the Latin modes read, value by value (inside a function: the maps die with it)."""
    settings = load_config(cfg)
    eager, mapped = _store(cfg, "eager"), _store(cfg, "mmap")
    names = sorted(set(mode_tables(settings, LATIN[0])) | set(mode_tables(settings, LATIN[1])))
    assert mapped.verify(names) == eager.verify(names) == {n: settings.tables[n].sha256 for n in names}
    count = 0
    for name in names:
        suffix = Path(settings.tables[name].path).suffix
        if suffix in (".npz", ".npy"):
            count += _compare_arrays(name, eager, mapped, suffix == ".npz")
    words = settings.languages["latin_toy"].word_list
    assert mapped.words(words) == eager.words(words)
    assert np.array_equal(mapped.matrix("toy_template_weights", "weights"),
                          eager.matrix("toy_template_weights", "weights"))
    assert mapped.weight_sets("toy_letter_sets", "wsets") == eager.weight_sets("toy_letter_sets", "wsets")
    return count


def test_the_mapped_store_returns_the_pinned_values() -> None:
    cfg, mm = _converted("ts_values")
    assert (mm / MANIFEST).is_file()
    count = _compare(cfg)
    gc.collect()  # Windows: a mapped file cannot be removed while a map is open
    assert count >= 40


def test_a_value_that_fits_no_allowed_width_stops_the_converter_and_writes_nothing() -> None:
    cfg = store_config("ts_overflow", "mmap", ("narrow_bits = [8, 16, 32, 64]", "narrow_bits = [8, 16, 32]"))
    done = convert_tables(cfg, "latin_max")  # the letter-chain keys reach 1.5e12: they need 64 bits
    assert done.returncode == 3, done.stderr[-800:]
    assert "toy_letters" in done.stderr.decode("utf-8", "replace")
    mm = WORK / "ts_overflow" / "mm"
    assert not mm.exists() or not any(mm.rglob("*")), "the converter wrote something before it refused"


def _rewrite(mm: Path, change: Callable[[dict[str, object]], None], reseal: bool) -> None:
    raw = json.loads((mm / MANIFEST).read_text(encoding="utf-8"))
    change(raw)
    if reseal:
        raw["self_sha256"] = seal(raw)
    (mm / MANIFEST).write_text(json.dumps(raw), encoding="utf-8")


def _biggest(mm: Path) -> Path:
    return max((p for p in mm.rglob("*.npy") if "bigram" in p.parent.name), key=lambda p: p.stat().st_size)


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def _garbage(path: Path) -> None:
    path.write_bytes(b"{not json")


def _foreign(raw: dict[str, object]) -> None:
    tables = raw["tables"]
    assert isinstance(tables, dict)
    tables["toy_bigram"]["source_sha256"] = "0" * 64


def _edited(raw: dict[str, object]) -> None:
    tables = raw["tables"]
    assert isinstance(tables, dict)
    tables["toy_bigram"]["source_bytes"] += 1


DAMAGE: dict[str, Callable[[Path], None]] = {
    "garbage": lambda mm: _garbage(mm / MANIFEST),
    "no_manifest": lambda mm: (mm / MANIFEST).unlink(),
    "edited_unsealed": lambda mm: _rewrite(mm, _edited, reseal=False),
    "foreign_source": lambda mm: _rewrite(mm, _foreign, reseal=True),
    "changed_file": lambda mm: _flip_last_byte(_biggest(mm)),
    "missing_file": lambda mm: _biggest(mm).unlink(),
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_a_damaged_or_foreign_derived_copy_is_refused_at_start(damage: str) -> None:
    cfg, mm = _converted(f"ts_damage_{damage}")
    DAMAGE[damage](mm)
    out = WORK / f"ts_damage_{damage}" / "out.hz"
    done = compress_file(SAMPLES[0], out, "latin_fast", "--no-fallback", "--config", str(cfg))
    assert done.returncode == 3, (done.returncode, done.stderr[-800:])
    assert not out.exists()


@pytest.mark.parametrize("mode", LATIN)
def test_both_stores_give_byte_identical_packages(mode: str) -> None:
    cfg, _ = _converted(f"ts_same_{mode}")
    eager_cfg = store_config(f"ts_same_{mode}_eager", "eager")
    out = WORK / f"ts_same_{mode}"
    for sample in SAMPLES:
        packages = []
        for config in (eager_cfg, cfg):
            pkg = out / f"{sample.stem}.{config.parent.name}.hz"
            done = compress_file(sample, pkg, mode, "--no-fallback", "--config", str(config))
            assert done.returncode == 0, done.stderr[-800:]
            packages.append(pkg.read_bytes())
        assert packages[0] == packages[1], sample.stem
        back = out / f"{sample.stem}.txt"
        done = decompress_file(out / f"{sample.stem}.{cfg.parent.name}.hz", back, "--config", str(cfg))
        assert done.returncode == 0 and back.read_bytes() == sample.read_bytes(), sample.stem
