"""The config subset (``ops/config_subset.py``): the browser build's config is MADE from the real one, never kept.

Why: a hand-trimmed copy drifts the day a row of ``fast`` changes. Proven here: the subset holds
exactly the sections of the asked modes (their tables, language, sub-sections), the ``--set`` rows and nothing
else; it loads; every kept mode signs like the real config; a typo in a ``--set`` row or a mode is refused; and
through the real tool in a child process the written config compresses the golden texts to the golden bytes.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib

import pytest

from hebzip.config.load import load_config
from hebzip.core.errors import ConfigError
from hebzip.ops.config_subset import make, split_sections
from hebzip.services.select import pinned_signature
from tests.support import CONFIG, ROOT, SAMPLE, TEXTS, golden_package, run_cli, work_dir

BROWSER_ROWS = {"engine": '"python"', "table_store": '"eager"', "fallback": "[]", "fallback_policy": '"on_error"',
                "signature_cache": '""', "verify_roundtrip": "false"}
"""The rows the website build sets (tools/site/build.toml says the same)."""


def test_the_fast_subset_keeps_exactly_fast_and_signs_like_the_real_config() -> None:
    text = make(CONFIG, ["fast"], BROWSER_ROWS)
    data = tomllib.loads(text)
    assert set(data) == {"hebzip", "tables", "languages", "modes"}
    assert set(data["modes"]) == {"fast"} and set(data["languages"]) == {"he"}
    assert set(data["tables"]) == {"he_words", "bigram_mc1", "template_m128", "template_m128_weights"}
    assert data["hebzip"]["engine"] == "python" and data["hebzip"]["fallback"] == []
    assert data["modes"]["fast"] == tomllib.loads(CONFIG.read_text(encoding="utf-8"))["modes"]["fast"]
    real = load_config(CONFIG)
    assert pinned_signature(real, "fast")[0] == 0x94


def test_sections_keep_their_comments_and_lose_nothing() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    sections = split_sections(text)
    assert "".join(line for s in sections for line in s.lines) == text
    assert sections[0].name == "" and sections[1].name == "hebzip"


@pytest.mark.parametrize(("modes", "rows", "word"), [(["fast"], {"enigne": '"python"'}, "enigne"),
                                                     (["fastt"], {}, "fastt")])
def test_a_typo_is_refused(modes: list[str], rows: dict[str, str], word: str) -> None:
    with pytest.raises(ConfigError, match=word):
        make(CONFIG, modes, rows)


def test_the_written_subset_compresses_to_the_golden_bytes() -> None:
    work = work_dir("subset_cli")
    out = work / "browser.toml"
    sets = [f"--set={row}={value}" for row, value in BROWSER_ROWS.items()]
    done = subprocess.run([sys.executable, "-m", "hebzip.ops.config_subset", "--modes", "fast", *sets,
                           "--out", str(out)], cwd=ROOT, capture_output=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-600:]
    for stem in SAMPLE:
        package = work / f"{stem}.hz"
        made = run_cli("compress", "--config", out, "--mode", "fast", "--force", TEXTS / f"{stem}.txt", package)
        assert made.returncode == 0, made.stderr[-600:]
        assert package.read_bytes() == golden_package("fast", stem), stem
