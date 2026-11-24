"""The language-pack table builder (the change-without-code rule): a new language = rows + tables made WITHOUT
the research prototype.

Why: the tables used to be copied from the research prototype, so "another country" needed prototype
code; the builder once could not make the template model's corpus prior (written empty) nor the
proclitic classes (refused by name). These tests run the real tool in a child process:
* built twice from the same corpus, the toy Latin tables are byte-identical (they are pinned by sha256)
  and equal to the ones the config pins - now with a REAL template prior and fitted start weights;
* the proclitic-class drawer can be switched on for the toy language with rows only: its two tables
  are built by the same tool (no code change) and the mode then round-trips through the real CLI - in
  both table stores (with ``mmap``: build -> paste rows -> convert -> compress, the builder names the step);
* a table without a recipe (given) is refused by name, never guessed;
* every recipe word of the config has a recipe function, every corpus format a reader.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from hebzip.config import schema as S
from hebzip.ops.corpus import READERS
from hebzip.ops.prefix import class_count, word_class
from hebzip.ops.recipes import RECIPES
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
TIMEOUT_S = 600
CONVERT_HINT = "python -m hebzip.ops.convert_tables"
"""The next step the builder's report must name (with ``mmap`` the new tables are not mapped until then)."""


def _build(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "hebzip.ops.build_langpack", "--config", str(config), "--language", "latin_toy",
           "--corpus", str(CORPUS), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S,
                          check=False)


def _rows(text: str) -> dict[str, str]:
    """``{table: sha256}`` from the tool's printed rows."""
    return {name: row["sha256"] for name, row in tomllib.loads(text)["tables"].items()}


def _config_into(name: str) -> Path:
    """The config with every toy table path moved into ``tests/_work/<name>/`` (the real files stay)."""
    work = work_dir(name)
    rel = work.relative_to(ROOT).as_posix()
    text = CONFIG.read_text(encoding="utf-8").replace('path = "tables/latin_toy/', f'path = "{rel}/')
    path = work / "hebzip.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_every_recipe_word_and_corpus_format_has_its_function() -> None:
    assert set(RECIPES) == set(S.RECIPE_ROWS)
    assert set(READERS) == {"lines", "jsonl"}


def test_building_twice_gives_the_pinned_tables() -> None:
    tables = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["tables"]
    pinned = {n: r["sha256"] for n, r in tables.items() if n.startswith("toy_")}
    first, second = _build(_config_into("langpack_a")), _build(_config_into("langpack_b"))
    assert first.returncode == 0, first.stderr[-1500:]
    assert second.returncode == 0, second.stderr[-1500:]
    assert _rows(first.stdout) == _rows(second.stdout) == pinned
    template = tables["toy_template"]["bytes"]
    assert template > 4096, "the toy template prior must have real entries (an empty one is 4018 bytes)"


def test_a_given_table_is_refused_by_name() -> None:
    cmd = [sys.executable, "-m", "hebzip.ops.build_langpack", "--language", "he", "--corpus", str(CORPUS),
           "--tables", "he_words", "--out", str(work_dir("langpack_given"))]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S,
                          check=False)
    assert done.returncode != 0 and "he_words" in done.stderr and "given" in done.stderr, done.stderr[-800:]


PX_ON = ('[modes.latin_max.predictors.prefix_class]\nkind = "null"', '''[modes.latin_max.predictors.prefix_class]
kind = "prefix_class"
table = "toy_prefix_classes"
prior_table = "toy_prefix_prior"
prior_add = 0.5
prior_k = 1.25
confidence_k = 729.0
trust0 = 0.1''')


def _px_tables(work: str, sha: dict[str, str]) -> str:
    out = ""
    for name, kind, rows in (("toy_prefix_classes", "prefix_classes",
                              'short_len = 2\ndouble_min_len = 4\ndouble_first = ""'),
                             ("toy_prefix_prior", "prefix_prior", "held_every = 0")):
        out += (f'\n[tables.{name}]\npath = "{work}/{name}.npy"\nsha256 = "{sha.get(name, "0" * 64)}"\nbytes = 0\n'
                f'\n[tables.{name}.recipe]\nkind = "{kind}"\n{rows}\n')
    return out


@pytest.mark.parametrize("store", STORES)
def test_proclitic_classes_switch_on_for_the_toy_language_with_rows_only(store: str) -> None:
    """The user's whole flow, in both stores: build -> paste the printed rows ->
    with ``mmap`` run the converter (the builder says so; before it, compress refuses naming it) -> compress
    and decompress. The derived copy goes to the test's own folder, never the real ``tables/mmap``."""
    name = f"langpack_px_{store}"
    work = work_dir(name)
    rel = work.relative_to(ROOT).as_posix()
    text = replaced(CONFIG.read_text(encoding="utf-8"), PX_ON, (STORE_ROW, f'table_store = "{store}"'),
                    (STORE_DIR_ROW, f'table_store_dir = "{rel}/mm"'))
    config = work / "hebzip.toml"
    config.write_text(text + _px_tables(rel, {}), encoding="utf-8")
    done = _build(config, "--tables", "toy_prefix_classes,toy_prefix_prior")
    assert done.returncode == 0, done.stderr[-1500:]
    assert CONVERT_HINT in done.stdout, done.stdout[-800:]
    config.write_text(text + _px_tables(rel, _rows(done.stdout)), encoding="utf-8")
    packed, back = work / "b.hz", work / "b.out"
    if store == "mmap":
        refused = compress_file(SAMPLE, packed, "latin_max", "--no-fallback", "--config", str(config))
        assert refused.returncode == 3 and b"convert_tables" in refused.stderr, refused.stderr[-800:]
        converted = convert_tables(config, "latin_max")
        assert converted.returncode == 0, converted.stderr[-800:]
    done_c = compress_file(SAMPLE, packed, "latin_max", "--no-fallback", "--config", str(config))
    assert done_c.returncode == 0, done_c.stderr[-1500:]
    assert decompress_file(packed, back, "--config", str(config)).returncode == 0
    assert back.read_bytes() == SAMPLE.read_bytes()


def test_the_hebrew_class_rule_is_the_experiments() -> None:
    """sp_model.prefix_class: <=2 letters -> 1; first letter one of the proclitics -> 2+i; vav + another
    proclitic in a word of >= 4 letters -> 9+i; else 0 (16 classes)."""
    lang = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["languages"]["he"]["prefixes"]
    rule = (lang, "ו", 2, 4)
    assert class_count(lang, "ו") == 16
    assert [word_class(w, *rule) for w in ("של", "ספר", "בית", "ובית", "והם", "ושלום", "לבית")] == [
        1, 0, 4, 11, 2, 15, 6]
