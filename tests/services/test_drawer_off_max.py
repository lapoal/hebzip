"""Mode max: switching any of its new drawers off is one word; the output changes, the round trip holds.

Design decision + the change-without-code rule, for every drawer of mode max: the five predictors (-> ``null``), the gap
classifier (-> ``none``), the word coder (-> ``direct``), the three letter inputs (-> ``null``) and the
letter mixer (-> ``word_only``). For each: a config where only that word says "plate", three texts
compressed and decompressed in SEPARATE processes with it. At least one payload must differ from the
real config's payload (the drawer really was in the path) and every text must come back exactly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.support import compress_file, config_variant, decompress_file, sample_paths, work_dir

WORKERS = 3
MODE = "max"


def _max(section: str, kind: str, plate: str) -> tuple[str, str]:
    """The kind row of one ``[modes.max.<section>]`` (the Latin pack's latin_max has rows with the same
    words in the same file, so a row is found by its section)."""
    head = f"[modes.max.{section}]\nkind = "
    return f'{head}"{kind}"', f'{head}"{plate}"'


OFF: dict[str, tuple[str, str]] = {
    "text_triples": _max("predictors.text_triples", "text_triples", "null"),
    "text_gap_starts": _max("predictors.text_gap_starts", "text_gap_starts", "null"),
    "corpus_gap_starts": _max("predictors.corpus_gap_starts", "corpus_gap_starts", "null"),
    "prefix_class": _max("predictors.prefix_class", "prefix_class", "null"),
    "recency": _max("predictors.recency", "recency", "null"),
    "gap": _max("gap", "punctuation", "none"),
    "word_coder": ('word_coder = "spelled"           #', 'word_coder = "direct"           #'),
    "cross_word": _max("letters.cross_word", "static_chain", "null"),
    "within_word": _max("letters.within_word", "static_chain", "null"),
    "text_letters": _max("letters.text_letters", "text_letters", "null"),
    "letter_mixer": _max("letter_mixer", "geometric", "word_only"),
}
"""Drawer -> the exact config text that switches it to its plate."""


def _round(text: Path, work: Path, cfg: Path | None) -> tuple[bytes, str | None]:
    pkg, back = work / f"{text.stem}.hz", work / f"{text.stem}.out"
    extra = ("--config", str(cfg)) if cfg else ()
    first = compress_file(text, pkg, MODE, "--no-fallback", *extra)
    if first.returncode != 0:
        return b"", f"{text.stem}: compress exit {first.returncode}: {first.stderr[-300:]!r}"
    second = decompress_file(pkg, back, *extra)
    if second.returncode != 0:
        return b"", f"{text.stem}: decompress exit {second.returncode}: {second.stderr[-300:]!r}"
    same = back.read_bytes() == text.read_bytes()
    return pkg.read_bytes(), None if same else f"{text.stem}: text differs after round trip"


def _run(name: str, cfg: Path | None) -> list[tuple[str, bytes, str | None]]:
    work = work_dir(name)
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda t: (t.stem, *_round(t, work, cfg)), sample_paths()))


@pytest.fixture(scope="module")
def default_packages() -> dict[str, bytes]:
    results = _run("off_max_default", None)
    assert not [err for _, _, err in results if err]
    return {stem: data for stem, data, _ in results}


@pytest.mark.parametrize("drawer", sorted(OFF))
def test_switching_a_max_drawer_off_changes_output_and_still_round_trips(drawer: str,
                                                                         default_packages: dict[str, bytes]) -> None:
    cfg = config_variant(f"off_max_cfg_{drawer}", OFF[drawer])
    results = _run(f"off_max_run_{drawer}", cfg)
    failures = [err for _, _, err in results if err]
    assert not failures, failures
    # payloads only: the header's signature byte changes with ANY row, the payload only if the drawer mattered
    changed = [stem for stem, data, _ in results if data[2:] != default_packages[stem][2:]]
    assert changed, f"{drawer}: the payload did not change"
