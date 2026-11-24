"""The max_plus table recipes through the real builder, on the toy Latin language - rows only.

* ``word_classes`` (k-means over ``--vectors``) + ``class_pairs`` (class of the word 1 / 2 back -> next word):
  switched on for latin_max with rows only, built twice -> the same sha256 (deterministic), every word gets a
  class, the class lists count every pair of the training documents, and the mode then round-trips through
  the real CLI in both table stores (build -> paste the rows -> convert under mmap -> compress).
* without ``--vectors`` the class recipe is refused by name, never guessed.
* ``spread_sets``: every listed weight is shared with its twin appended at the end (no mode of the real
  config uses it today, so the table row is added to a copy of the config: the recipe stays a tested drawer).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tests.support import (
    CONFIG,
    ROOT,
    STORE_DIR_ROW,
    STORE_ROW,
    STORES,
    compress_file,
    convert_tables,
    decompress_file,
    replaced,
    work_dir,
)

CORPUS = ROOT / "langpacks" / "latin_toy" / "corpus.txt"
SAMPLE = ROOT / "langpacks" / "latin_toy" / "samples" / "bridge.txt"
WORDS = ROOT / "tables" / "latin_toy" / "words.txt"
TIMEOUT_S = 600
K = 4
CHANNELS = ('            "corpus_gap_starts", "prefix_class", "recency"]\n\n[modes.latin_max.gap]',
            '            "corpus_gap_starts", "prefix_class", "recency", "cls1", "cls2"]\n\n[modes.latin_max.gap]')
SPELLER = "[modes.latin_max.speller]"


def _predictors(rel: str) -> str:
    out = ""
    for name, window in (("cls1", 1), ("cls2", 2)):
        out += (f'[modes.latin_max.predictors.{name}]\nkind = "corpus_classes"\ntable = "toy_classes"\n'
                f'prior_table = "toy_class_pairs"\nwindow = {window}\ndiscount = 0.01171875\ntrust0 = 0.2\n\n')
    return out


def _tables(rel: str, sha: dict[str, str]) -> str:
    return (f'\n[tables.toy_classes]\npath = "{rel}/classes.npy"\nsha256 = "{sha.get("toy_classes", "0" * 64)}"\n'
            f'bytes = 0\n\n[tables.toy_classes.recipe]\nkind = "word_classes"\nclasses = {K}\nfit_words = 200\n'
            f'iterations = 5\nseed = 7\n'
            f'\n[tables.toy_class_pairs]\npath = "{rel}/class_pairs.npz"\n'
            f'sha256 = "{sha.get("toy_class_pairs", "0" * 64)}"\nbytes = 0\n\n[tables.toy_class_pairs.recipe]\n'
            f'kind = "class_pairs"\nheld_every = 0\nmin_count = 1\n')


def _setup(name: str, store: str) -> tuple[Path, str, str, Path]:
    work = work_dir(name)
    rel = work.relative_to(ROOT).as_posix()
    vectors = work / "vectors.npy"
    n = len(WORDS.read_text(encoding="utf-8").split("\n")) - 1
    np.save(vectors, np.random.default_rng(3).standard_normal((n, 8)).astype(np.float16))
    text = replaced(CONFIG.read_text(encoding="utf-8"), CHANNELS, (SPELLER, _predictors(rel) + SPELLER),
                    (STORE_ROW, f'table_store = "{store}"'), (STORE_DIR_ROW, f'table_store_dir = "{rel}/mm"'))
    return work, rel, text, vectors


def _build(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "hebzip.ops.build_langpack", "--config", str(config), "--language", "latin_toy",
           "--corpus", str(CORPUS), "--tables", "toy_classes,toy_class_pairs", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S,
                          check=False)


def _rows(text: str) -> dict[str, str]:
    return {name: row["sha256"] for name, row in tomllib.loads(text)["tables"].items()}


def _pinned_pairs() -> int:
    """Every pair of neighbouring words in the toy training documents, as the pinned ``toy_bigram`` counted
    them with the real tokenizer (its recipe also holds nothing out): the sum of its ``ctx_n``."""
    with np.load(ROOT / "tables" / "latin_toy" / "bigram.npz") as z:
        return int(z["ctx_n"].sum())


@pytest.mark.parametrize("store", STORES)
def test_word_classes_switch_on_for_the_toy_language_with_rows_only(store: str) -> None:
    work, rel, text, vectors = _setup(f"plus_build_{store}", store)
    config = work / "hebzip.toml"
    config.write_text(text + _tables(rel, {}), encoding="utf-8")
    first = _build(config, "--vectors", str(vectors))
    second = _build(config, "--vectors", str(vectors))
    assert first.returncode == 0, first.stderr[-1500:]
    assert _rows(first.stdout) == _rows(second.stdout), "the same corpus and vectors give the same tables"
    classes = np.load(work / "classes.npy")
    lists = np.load(work / "class_pairs.npz")
    assert classes.min() >= 0 and classes.max() < K and len(np.unique(classes)) > 1
    assert int(lists["N1"].sum()) == _pinned_pairs(), "window 1: every training pair is counted once"
    assert 0 < int(lists["N2"].sum()) < int(lists["N1"].sum()), "window 2: one pair fewer per document"
    for w in (1, 2):
        assert int(lists[f"off{w}"][-1]) == len(lists[f"ids{w}"])
    config.write_text(text + _tables(rel, _rows(first.stdout)), encoding="utf-8")
    if store == "mmap":
        assert convert_tables(config, "latin_max").returncode == 0
    packed, back = work / "b.hz", work / "b.out"
    done = compress_file(SAMPLE, packed, "latin_max", "--no-fallback", "--config", str(config))
    assert done.returncode == 0, done.stderr[-1500:]
    assert decompress_file(packed, back, "--config", str(config)).returncode == 0
    assert back.read_bytes() == SAMPLE.read_bytes()


def test_the_class_recipe_without_vectors_is_refused_by_name() -> None:
    work, rel, text, _ = _setup("plus_build_novec", "eager")
    config = work / "hebzip.toml"
    config.write_text(text + _tables(rel, {}), encoding="utf-8")
    done = _build(config)
    assert done.returncode == 2 and "--vectors" in done.stderr and "toy_classes" in done.stderr, done.stderr[-800:]


SPREAD_ROWS = """
[tables.spread_probe]
path = "tables/plus/mix_letter_sets.json"
sha256 = "%s"
bytes = 0

[tables.spread_probe.recipe]
kind = "spread_sets"
source = "letter_start_sets"
halve = [1, 2]
""" % ("0" * 64)


def test_spread_sets_share_each_listed_weight_with_its_twin() -> None:
    work = work_dir("plus_build_spread")
    config = work / "hebzip.toml"
    reads = CONFIG.read_text(encoding="utf-8").replace('start_sets = "letter_start_sets"',
                                                     'start_sets = "spread_probe"', 1)
    config.write_text(reads + SPREAD_ROWS, encoding="utf-8")  # the builder builds only what a mode reads
    cmd = [sys.executable, "-m", "hebzip.ops.build_langpack", "--config", str(config), "--language", "he",
           "--corpus", str(CORPUS), "--tables", "spread_probe", "--out", str(work)]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S,
                          check=False)
    assert done.returncode == 0, done.stderr[-800:]
    source = json.loads((ROOT / "tables" / "letter_start_sets.json").read_text(encoding="utf-8"))["wsets"]
    made = json.loads((work / "mix_letter_sets.json").read_text(encoding="utf-8"))["wsets"]
    assert set(made) == set(source)
    for key, w in source.items():
        assert made[key] == [w[0], w[1] / 2, w[2] / 2, w[3], w[1] / 2, w[2] / 2]
    assert "spread_probe" in _rows(done.stdout)
