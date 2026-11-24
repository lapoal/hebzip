"""The website build: the site is the code (no drift), its config signs like the real one, it reaches no other
origin, and its data carries numbers only.

Why these and not more: the browser run itself is checked in a real headless browser (see tools/site/README.md);
what can break silently between such runs is (1) a code / table / bench change without a rebuild, (2) a
hand edit that adds a request to another site, (3) a machine path or a text leaking into the published data.
Run: python tools/site/run_tests.py (cwd-independent).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import build_site
import pytest
from bench_data import bench_for_site
from table_files import read_table, unpack

from hebzip.config.load import DEFAULT_CONFIG, load_config, load_dict
from hebzip.services.select import pinned_signature

SITE = build_site.REPO / "site"
PY = sys.executable


def test_the_site_is_a_fresh_build_of_the_repository() -> None:
    done = subprocess.run([PY, str(build_site.HERE / "build_site.py"), "--check"], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_browser_config_signs_every_mode_like_the_real_config() -> None:
    small = load_dict(tomllib.loads((SITE / "py" / "browser.toml").read_text(encoding="utf-8")))
    real = load_config(DEFAULT_CONFIG)
    for mode in ("fast", "en_fast"):
        assert pinned_signature(small, mode) == pinned_signature(real, mode), mode
    assert small.hebzip.engine == "python" and small.hebzip.fallback == [] and set(small.modes) == {"fast", "en_fast"}


def test_every_fetched_file_is_there_and_pinned() -> None:
    manifest = json.loads((SITE / "build.json").read_text(encoding="utf-8"))
    for name, row in manifest["files"].items():
        assert hashlib.sha256(read_table(SITE, name)).hexdigest() == row["sha256"], name
        assert (SITE / row.get("file", name)).stat().st_size == row["bytes"], name
    assert set(manifest["packs"]) == {"fast", "en_fast"}
    assert all(name in manifest["files"] for pack in manifest["packs"].values() for name in pack)


def test_the_website_tables_unpack_to_the_pinned_files(tmp_path: Path) -> None:
    real = load_config(DEFAULT_CONFIG)
    names = ("he_words", "en_bigram")
    written = unpack(SITE, {n: (real.tables[n].path, real.tables[n].sha256) for n in names}, tmp_path)
    for name, path in zip(names, written, strict=True):
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == real.tables[name].sha256
    with pytest.raises(ValueError, match="not the pinned"):
        unpack(SITE, {"x": (real.tables["he_words"].path, "0" * 64)}, tmp_path)


def test_the_page_reaches_no_other_origin() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert "connect-src 'self'" in html and "default-src 'self'" in html
    assert not re.search(r"""(src|href)=["'](https?:)?//""", html), "a resource from another site"
    for js in (SITE / "assets").glob("*.js"):
        urls = re.findall(r"https?://[^\s\"'`]+", js.read_text(encoding="utf-8"))
        assert all(u.startswith("https://github.com/${") for u in urls), (js.name, urls)


def test_the_share_link_carries_numbers_only() -> None:
    share = (SITE / "assets" / "share.js").read_text(encoding="utf-8")
    assert "result.text" not in share and ".value" not in share


def test_published_data_holds_no_machine_path() -> None:
    for name in ("build.json", "data/bench.json", "py/browser.toml", "settings.json"):
        text = (SITE / name).read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\\\|[A-Za-z]:/|/Users/|/home/", text), name


@pytest.mark.parametrize("cold", ["fast_cli", "en_max_cli"])
def test_bench_data_folds_the_cold_rows_and_keeps_the_numbers(cold: str) -> None:
    results = build_site.REPO / "tools" / "compare_page" / "bench" / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    data = json.loads((results / "bench.json").read_text(encoding="utf-8"))
    out = bench_for_site(summary, data)
    rows = {r["id"]: r for lang in out["languages"].values() for r in lang["rows"]}
    assert cold not in rows and rows[cold.removesuffix("_cli")]["cold_compress_s"] is not None
    assert rows["fast"]["prior_bytes"] == summary["prior_table_bytes"]["fast"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", out["recorded"])
