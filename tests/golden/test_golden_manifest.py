"""The golden reference is intact: every file listed in ``golden/manifest.json`` has its sha256, and the
manifest lists exactly the golden folder - no package missing, none extra, every mode of the config present.

Why: ``golden/`` is the read-only yardstick; if a byte of it changed (or a package went missing, or a stray
one appeared), every identity test would measure against the wrong thing. The set is made by
``python -m tests.golden.make_golden``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tests.support import GOLDEN, GOLDEN_MODES, HEBREW_TEXTS, ROOT, SIGNATURE_BYTE, mode_texts, real_config


def _manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))
    return data


def test_every_golden_file_matches_the_manifest() -> None:
    manifest = _manifest()
    for name, row in manifest["texts"].items():
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"], name
    for mode, entry in manifest["modes"].items():
        for name, row in entry["packages"].items():
            data = (GOLDEN / mode / f"{name}.hz").read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["sha256"], f"{mode}/{name}"


def test_the_manifest_covers_every_mode_and_text_and_nothing_else() -> None:
    manifest, modes = _manifest(), real_config()["modes"]
    assert set(manifest["modes"]) == set(GOLDEN_MODES) == set(modes) == set(SIGNATURE_BYTE)
    assert len(list((GOLDEN / "texts").glob("*.txt"))) == HEBREW_TEXTS
    for mode, entry in manifest["modes"].items():
        on_disk = sorted(p.stem for p in (GOLDEN / mode).iterdir())
        assert on_disk == sorted(entry["packages"]) == [t.stem for t in mode_texts(mode)], mode
        assert entry["id"] == modes[mode]["id"] and entry["signature_byte"] == SIGNATURE_BYTE[mode], mode
