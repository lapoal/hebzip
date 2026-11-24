"""The max_plus drawers and the ``choose`` mode kind through the real CLI: separate processes,
exact round trips.

1. A ``choose`` mode keeps the SMALLER of its candidates' packages: the package equals the candidate's own
   package byte for byte (header = the candidate's id), and decodes in another process with nothing but the
   header - a choose mode never appears in a package. No mode of the real config is a choose mode today, so
   the test adds one, ``pick`` = fast or max, to a copy of the config (the drawer stays alive and tested).
2. Every max_plus drawer switched to its plate is ONE word (the change-without-code rule): the payload
   changes (the drawer was in the path) and every text still comes back exactly.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.support import CONFIG, compress_file, decompress_file, sample_paths, work_dir

WORKERS = 3
CANDIDATES = {"fast": 1, "max": 2}
PICK = '\n[modes.pick]\nid = 9\nkind = "choose"\ncandidates = ["fast", "max"]\n'
"""The test-only choose mode (same as ``tests/config/test_load_plus.PICK``)."""
Packs = dict[str, tuple[bytes, str | None]]


def _pack(text: Path, work: Path, mode: str, cfg: Path | None = None) -> tuple[bytes, str | None]:
    """Compress, then decompress in another process, both with ``cfg`` (default: the real config)."""
    pkg, back = work / f"{text.stem}.{mode}.hz", work / f"{text.stem}.{mode}.out"
    extra = ("--config", str(cfg)) if cfg else ()
    first = compress_file(text, pkg, mode, "--no-fallback", *extra)
    if first.returncode != 0:
        return b"", f"{text.stem}/{mode}: compress exit {first.returncode}: {first.stderr[-300:]!r}"
    second = decompress_file(pkg, back, *extra)
    if second.returncode != 0:
        return b"", f"{text.stem}/{mode}: decompress exit {second.returncode}: {second.stderr[-300:]!r}"
    return pkg.read_bytes(), None if back.read_bytes() == text.read_bytes() else f"{text.stem}/{mode}: differs"


def _all(name: str, mode: str, cfg: Path | None = None) -> Packs:
    work = work_dir(name)
    with ThreadPoolExecutor(WORKERS) as pool:
        return dict(pool.map(lambda t: (t.stem, _pack(t, work, mode, cfg)), sample_paths()))


def test_choose_keeps_the_smaller_candidate_package() -> None:
    candidates = {m: _all(f"plus_{m}", m) for m in CANDIDATES}
    cfg = work_dir("plus_pick_cfg") / "hebzip.toml"
    cfg.write_text(CONFIG.read_text(encoding="utf-8") + PICK, encoding="utf-8")
    chosen: dict[str, bytes] = {}
    work = work_dir("plus_choose")
    for text in sample_paths():
        pkg, back = work / f"{text.stem}.hz", work / f"{text.stem}.out"
        assert compress_file(text, pkg, "pick", "--no-fallback", "--config", str(cfg)).returncode == 0
        assert decompress_file(pkg, back).returncode == 0, "the real config (no pick mode) opens it"
        assert back.read_bytes() == text.read_bytes()
        chosen[text.stem] = pkg.read_bytes()
    errors = [e for got in candidates.values() for _, e in got.values() if e]
    assert not errors, errors
    for stem, data in chosen.items():
        sizes = {m: len(candidates[m][stem][0]) for m in CANDIDATES}
        best = min(CANDIDATES, key=lambda m: (sizes[m], CANDIDATES[m]))
        assert data == candidates[best][stem][0], f"{stem}: not the smaller candidate {sizes}"
        assert data[0] & 0x7F == CANDIDATES[best], "the header names the candidate, not the choose mode"


def plate_config(section: str, plate: str) -> Path:
    """A config where the drawer of ``[section]`` says its plate word (its rows may stay - a plate reads none)."""
    text = CONFIG.read_text(encoding="utf-8")
    pattern = re.compile(r"(\[" + re.escape(section) + r"\][^\n]*\nkind = )\"[a-z_]+\"")
    assert len(pattern.findall(text)) == 1, section
    path = work_dir(f"plus_cfg_{section}") / "hebzip.toml"
    path.write_text(pattern.sub(lambda m: m.group(1) + f'"{plate}"', text), encoding="utf-8")
    return path


OFF = {
    "class_pairs": ("modes.max_plus.predictors.class_pairs", "null"),
    "class_skip": ("modes.max_plus.predictors.class_skip", "null"),
    "topic": ("modes.max_plus.predictors.topic", "null"),
    "mixer": ("modes.max_plus.mixer", "static"),
}
"""Drawer -> (its section, its plate word)."""


@pytest.fixture(scope="module")
def plus() -> Packs:
    return _all("plus_max_plus", "max_plus")


@pytest.mark.parametrize("drawer", sorted(OFF))
def test_switching_a_plus_drawer_off_changes_output_and_still_round_trips(drawer: str, plus: Packs) -> None:
    section, plate = OFF[drawer]
    got = _all(f"plus_off_{drawer}", "max_plus", plate_config(section, plate))
    errors = [e for _, e in [*got.values(), *plus.values()] if e]
    assert not errors, errors
    changed = [s for s, (data, _) in got.items() if data[2:] != plus[s][0][2:]]
    assert changed, f"{drawer}: the payload did not change"
