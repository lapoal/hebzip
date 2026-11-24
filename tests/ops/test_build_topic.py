"""The topic table recipes through the real builder.

* ``topic_neighbours`` (``ops/topics.py``): on random vectors the lists equal a brute-force reference (the
  ``neighbours`` most similar ids in ``skip_top .. candidates - 1``, self excluded, cosine >= floor), each slice
  sorted by id, weights 1..levels rising with the cosine, every other word without a list.
* switched on for the toy language with ROWS only (a ``text_topic`` channel on latin_max): built twice -> the
  same sha256, then the mode round-trips through the real CLI in both table stores.
* ``append_sets``: every vector of the source gets the ``append`` values at its end; the pinned Hebrew set of
  ``max_plus`` (``topic_trust_sets``) is exactly that.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from hebzip.ops.topics import neighbour_lists
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
CHANNELS = ('            "corpus_gap_starts", "prefix_class", "recency"]\n\n[modes.latin_max.gap]',
            '            "corpus_gap_starts", "prefix_class", "recency", "topic"]\n\n[modes.latin_max.gap]')
SPELLER = "[modes.latin_max.speller]"
TOPIC = ('[modes.latin_max.predictors.topic]\nkind = "text_topic"\ntable = "toy_topic"\nwindow = 8\n'
         'confidence_k = 4.0\ntrust0 = 0.1\n\n')


def _reference(v: np.ndarray, skip: int, cand: int, k: int, floor: float, levels: int) -> list[dict[int, int]]:
    u = v.astype(np.float64)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    out: list[dict[int, int]] = [{} for _ in range(len(v))]
    for y in range(skip, cand):
        sims = {x: float(u[y] @ u[x]) for x in range(skip, cand) if x != y}
        best = sorted(sims, key=lambda x: (-sims[x], x))[:k]
        out[y] = {x: max(1, min(levels, round(levels * (sims[x] - floor) / (1 - floor))))
                  for x in best if sims[x] >= floor}
    return out


def test_neighbour_lists_equal_a_brute_force_reference() -> None:
    v = np.random.default_rng(11).standard_normal((60, 6)).astype(np.float16)
    made = neighbour_lists(v, 5, 50, 7, 0.2, 255)
    off, ids, wt = made["off"], made["ids"], made["wt"]
    assert len(off) == 61 and int(off[-1]) == len(ids) == len(wt)
    ref = _reference(v, 5, 50, 7, 0.2, 255)
    got = []
    for y in range(60):
        sl = slice(int(off[y]), int(off[y + 1]))
        assert ids[sl].tolist() == sorted(ids[sl].tolist())
        got.append(dict(zip(ids[sl].tolist(), wt[sl].tolist(), strict=True)))
    assert [set(g) for g in got] == [set(r) for r in ref], "the same neighbours"
    diffs = [abs(g[x] - r[x]) for g, r in zip(got, ref, strict=True) for x in r]
    assert max(diffs) <= 1, "float32 vs float64 cosine: a weight may differ by one rounding step"
    assert all(not got[y] for y in (*range(5), *range(50, 60))), "outside skip_top..candidates: no list"


def _setup(name: str, store: str) -> tuple[Path, str, str, Path]:
    work = work_dir(name)
    rel = work.relative_to(ROOT).as_posix()
    vectors = work / "vectors.npy"
    n = len(WORDS.read_text(encoding="utf-8").split("\n")) - 1
    np.save(vectors, np.random.default_rng(3).standard_normal((n, 8)).astype(np.float16))
    text = replaced(CONFIG.read_text(encoding="utf-8"), CHANNELS, (SPELLER, TOPIC + SPELLER),
                    (STORE_ROW, f'table_store = "{store}"'), (STORE_DIR_ROW, f'table_store_dir = "{rel}/mm"'))
    return work, rel, text, vectors


def _table(rel: str, sha: str) -> str:
    return (f'\n[tables.toy_topic]\npath = "{rel}/topic.npz"\nsha256 = "{sha}"\nbytes = 0\n\n'
            '[tables.toy_topic.recipe]\nkind = "topic_neighbours"\nneighbours = 6\nskip_top = 20\n'
            'candidates = 400\nsim_floor = 0.3\nlevels = 255\n')


def _build(config: Path, tables: str, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "hebzip.ops.build_langpack", "--config", str(config), "--corpus", str(CORPUS),
           "--tables", tables, *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S,
                          check=False)


def _rows(text: str) -> dict[str, str]:
    return {name: row["sha256"] for name, row in tomllib.loads(text)["tables"].items()}


@pytest.mark.parametrize("store", STORES)
def test_the_topic_channel_switches_on_for_the_toy_language_with_rows_only(store: str) -> None:
    work, rel, text, vectors = _setup(f"topic_build_{store}", store)
    config = work / "hebzip.toml"
    config.write_text(text + _table(rel, "0" * 64), encoding="utf-8")
    first = _build(config, "toy_topic", "--language", "latin_toy", "--vectors", str(vectors))
    second = _build(config, "toy_topic", "--language", "latin_toy", "--vectors", str(vectors))
    assert first.returncode == 0, first.stderr[-1500:]
    assert _rows(first.stdout) == _rows(second.stdout), "the same vectors give the same table"
    with np.load(work / "topic.npz") as z:
        assert int(z["off"][20]) == 0 and int(z["off"][-1]) == len(z["ids"]) > 0
    config.write_text(text + _table(rel, _rows(first.stdout)["toy_topic"]), encoding="utf-8")
    if store == "mmap":
        assert convert_tables(config, "latin_max").returncode == 0
    packed, back = work / "b.hz", work / "b.out"
    done = compress_file(SAMPLE, packed, "latin_max", "--no-fallback", "--config", str(config))
    assert done.returncode == 0, done.stderr[-1500:]
    assert decompress_file(packed, back, "--config", str(config)).returncode == 0
    assert back.read_bytes() == SAMPLE.read_bytes()


def test_append_sets_add_the_new_channels_start_trust_to_every_vector() -> None:
    work = work_dir("topic_build_append")
    done = _build(CONFIG, "topic_trust_sets", "--language", "he", "--out", str(work))
    assert done.returncode == 0, done.stderr[-800:]
    pinned = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["tables"]
    for made_name, source, table in (("trust_sets.json", "trust_sets_wiki", "topic_trust_sets"),):
        tail = pinned[table]["recipe"]["append"]
        src = json.loads((ROOT / pinned[source]["path"]).read_text(encoding="utf-8"))["wsets"]
        made = json.loads((work / made_name).read_text(encoding="utf-8"))["wsets"]
        assert made == {k: v + tail for k, v in src.items()}
        assert _rows(done.stdout)[table] == pinned[table]["sha256"]
