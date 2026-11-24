"""The public measurement (``atar.bench``): config, manifest, dataset rebuild, report rules, a real run.

Real path (the real-path rule): the run test calls the same ``main`` as ``python -m atar.bench run``, with
a copy of ``config/bench.toml`` whose folders point into ``tests/_work``, on the shipped Hebrew Wikipedia
texts, through the real drawers (gzip, zstd, PPMd). The report rules are checked on numbers written
here: a total exists only when every text came back exact, and a "seen" text never enters a clean group.
"""

from __future__ import annotations

import json
import tomllib
from typing import Any

import pytest

from atar.bench import cli, report, run
from atar.bench.config import DEFAULT_BENCH, load, parse
from atar.bench.datasets import omilab_line
from atar.bench.sets import TextRef, load_manifest, text_bytes
from atar.compressors.registry import build
from atar.config.load import REPO, ROOT
from atar.core.errors import ConfigError
from tests.support import WORK

MANIFEST = ROOT / "bench" / "manifest.json"


def _bench() -> dict[str, Any]:
    return tomllib.loads(DEFAULT_BENCH.read_text(encoding="utf-8"))


def test_the_real_bench_config_loads_and_every_row_is_a_page_row() -> None:
    settings = load(DEFAULT_BENCH, {})
    ids = {r.tool.id for r in settings.rows}
    assert {"gzip", "xz", "bz2", "zstd", "brotli", "ppmd", "fast", "max", "max_plus", "fast_cli"} <= ids
    assert all(r.tool.ours == bool(r.hebzip_mode) for r in settings.rows)


def test_an_unknown_row_field_is_refused() -> None:
    data = _bench()
    data["rows"][0]["colour"] = "red"
    with pytest.raises(ConfigError, match="colour"):
        parse(data, {})


def test_our_row_needs_its_mode() -> None:
    data = _bench()
    ours = next(r for r in data["rows"] if r["ours"])
    del ours["hebzip_mode"]
    with pytest.raises(ConfigError, match="hebzip_mode"):
        parse(data, {})


def test_every_shipped_text_is_the_pinned_one() -> None:
    manifest = load_manifest(MANIFEST)
    shipped = [t for s in manifest.sets for t in s.texts if t.file]
    assert len(shipped) >= 40
    for ref in shipped:
        assert len(text_bytes(ref, None)) == ref.bytes


def test_a_changed_text_is_refused_not_measured() -> None:
    ref = next(t for s in load_manifest(MANIFEST).sets for t in s.texts if t.file)
    wrong = TextRef(ref.set_id, ref.name, ref.language, ref.length, "0" * 64, ref.bytes, ref.file, "", {})
    with pytest.raises(ConfigError, match="sha256"):
        text_bytes(wrong, None)


def test_omilab_line_cuts_the_label_and_undoes_the_tokenisation() -> None:
    assert omilab_line("שלום , עולם ( טוב ) .\t1") == "שלום, עולם (טוב)."


def _data(rows: list[tuple[str, str, int, bool]]) -> dict[str, Any]:
    tools = {"a": {"ours": True, "languages": ["he"], "header_bytes": 2, "hebzip_mode": "m", "version": "v"},
             "b": {"ours": False, "languages": ["he"], "header_bytes": 10, "hebzip_mode": "", "version": "v"}}
    return {"tools": tools, "rows": [{"tool": t, "text": x, "raw": 100, "bytes": b, "exact": e,
                                      "package_sha256": t + x, "compress_s": 0.1, "decompress_s": 0.1}
                                     for t, x, b, e in rows]}


def _group(summary: dict[str, Any], key: str) -> dict[str, Any]:
    found: dict[str, Any] = next(g for g in summary["groups"] if g["group"] == key)
    return found


def test_results_never_record_this_machines_paths() -> None:
    settings = load(DEFAULT_BENCH, {})
    row = next(r for r in settings.rows if r.tool.id == "fast_cli")
    info = run._tool_info(row, build(row.tool))
    assert info["repo"] == "" and info["python"] == ""
    assert run._place("D:/other/copy", "C:/this") == "D:/other/copy"


def test_report_totals_only_complete_exact_groups_and_keeps_seen_texts_apart() -> None:
    manifest = load_manifest(MANIFEST)
    names = [t.name for t in manifest.sets[0].texts]
    rows = [("a", n, 50, True) for n in names] + [("b", n, 60, True) for n in names[1:]]
    rows.append(("b", names[0], 61, False))
    seen = {"texts": {names[0]: {"title_in_corpus": True}}}
    summary = report.summary(_data(rows), manifest, seen, 0.05)
    clean = _group(summary, manifest.sets[0].id)
    assert clean["texts"] == len(names) - 1 and names[0] not in clean["names"]
    assert clean["tools"]["a"]["as_is"] == 50 * (len(names) - 1)
    assert clean["tools"]["b"]["pct_vs_ours"] == 20.0
    assert clean["tools"]["b"]["no_header"] == (60 - 10) * (len(names) - 1)
    dirty = _group(summary, f"{manifest.sets[0].id} (seen in training)")
    assert dirty["tools"]["b"]["as_is"] is None, "a text that did not come back exact must not be summed"


def test_a_real_run_measures_exactly_and_leaves_no_text_behind() -> None:
    config = WORK / "bench.toml"
    text = DEFAULT_BENCH.read_text(encoding="utf-8")
    for row in ('results = "bench/results"', 'work = "material/bench"'):
        key = row.split(" ")[0]
        assert row in text
        text = text.replace(row, f'{key} = "{(WORK / key).as_posix()}"')
    config.write_text(text, encoding="utf-8")
    assert cli.main(["run", "--config", str(config), "--rows", "gzip,zstd,ppmd", "--languages", "he"]) == 0
    data = json.loads((WORK / "results" / "bench.json").read_text(encoding="utf-8"))
    measured = {(r["tool"], r["set"]) for r in data["rows"]}
    assert ("zstd", "he_wiki") in measured and ("ppmd", "he_own") in measured
    assert all(r["exact"] and r["status"] == "ok" for r in data["rows"])
    assert not (WORK / "work").exists(), "the work folder must be removed at the end"
    sample = (REPO / "golden" / "texts" / "wiki_cat.txt").read_text(encoding="utf-8").split()[5:12]
    for path in (WORK / "results").iterdir():
        assert " ".join(sample) not in path.read_text(encoding="utf-8"), f"text found in {path.name}"
    sets = {s for _, s in measured}
    assert "he_everyday_omilab" not in sets, "a set whose dataset was not given is skipped"
