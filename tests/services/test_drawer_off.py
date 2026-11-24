"""Turning a drawer off is one word in the config: the output changes, the round trip still holds.

Design decision (every predictor is a drawer with a plate) + the change-without-code rule ("a method = a word in the
table that swaps a drawer"). For each drawer of mode fast: a config where only that word says
"plate", three texts compressed and decompressed in SEPARATE processes with that config. At least
one payload must differ from the default config's payload (the drawer really was in the path), and
every text must come back byte for byte.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.support import compress_file, config_variant, decompress_file, sample_paths, work_dir

WORKERS = 4


def _fast(section: str, kind: str, plate: str) -> tuple[str, str]:
    """The kind row of one ``[modes.fast.<section>]`` (mode max has rows with the same words)."""
    head = f"[modes.fast.{section}]\nkind = "
    return f'{head}"{kind}"', f'{head}"{plate}"'


OFF: dict[str, tuple[tuple[str, str], ...]] = {
    "corpus_pairs": (_fast("predictors.corpus_pairs", "corpus_pairs", "null"),),
    "text_pairs": (_fast("predictors.text_pairs", "text_pairs", "null"),),
    "text_counts": (_fast("predictors.text_counts", "text_counts", "null"),),
    "base": (_fast("base", "unigram_zipf", "uniform"),),
    "mixer": (_fast("mixer", "trust", "static"),),
    "template": (('kind = "cm"   ', 'kind = "flat" '),),
    "tokenizer": (('tokenizer = "overlay"            #', 'tokenizer = "literal"            #'),),
    "dictionary": (('dictionary = "word_list"         #', 'dictionary = "null"         #'),
                   ('word_model = "mixture"           #', 'word_model = "null"           #')),
}
"""Drawer -> the exact config text that switches it (and, for the dictionary, what needs it) off."""


def _default_packages() -> dict[str, bytes]:
    work = work_dir("off_default")
    out: dict[str, bytes] = {}
    for text in sample_paths():
        done = compress_file(text, work / f"{text.stem}.hz", "fast", "--no-fallback")
        assert done.returncode == 0, done.stderr[-300:]
        out[text.stem] = (work / f"{text.stem}.hz").read_bytes()
    return out


def _round(text: Path, work: Path, cfg: Path) -> tuple[bytes, str | None]:
    pkg, back = work / f"{text.stem}.hz", work / f"{text.stem}.out"
    first = compress_file(text, pkg, "fast", "--no-fallback", "--config", str(cfg))
    if first.returncode != 0:
        return b"", f"{text.stem}: compress exit {first.returncode}: {first.stderr[-300:]!r}"
    second = decompress_file(pkg, back, "--config", str(cfg))
    if second.returncode != 0:
        return b"", f"{text.stem}: decompress exit {second.returncode}: {second.stderr[-300:]!r}"
    same = back.read_bytes() == text.read_bytes()
    return pkg.read_bytes(), None if same else f"{text.stem}: text differs after round trip"


@pytest.fixture(scope="module")
def default_packages() -> dict[str, bytes]:
    return _default_packages()


@pytest.mark.parametrize("drawer", sorted(OFF))
def test_switching_a_drawer_off_changes_output_and_still_round_trips(drawer: str,
                                                                     default_packages: dict[str, bytes]) -> None:
    cfg = config_variant(f"off_cfg_{drawer}", *OFF[drawer])
    work = work_dir(f"off_run_{drawer}")
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda t: (t.stem, *_round(t, work, cfg)), sample_paths()))
    failures = [err for _, _, err in results if err]
    assert not failures, failures
    # payloads only: the header's signature byte changes with ANY row, the payload only if the drawer mattered
    changed = [stem for stem, data, _ in results if data[2:] != default_packages[stem][2:]]
    assert changed, f"{drawer}: the payload did not change"
