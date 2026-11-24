"""Make the golden set: every mode x its language's texts, packaged by the real CLI, proven, pinned.

    python -m tests.golden.make_golden

Why a tool and not a test: the golden packages are the yardstick the identity
tests compare against - they must be made ONCE, on purpose, after a deliberate format change (new rows,
new tables), never silently by a test run. Each package is checked before it is written:
1. compressed by the real CLI with the real config (``numba`` engine), ``--no-fallback``;
2. compressed again with the ``python`` engine - the bytes must be identical (one kernel source);
3. decompressed in ANOTHER process - the text must come back byte for byte.
Then ``golden/manifest.json`` records the sha256 and size of every text and package (and the Python /
numpy versions, since float maths of the max modes is tied to the platform). Old packages of a mode are
removed first, so the folder holds exactly the current set. Scratch files: ``tests/_work`` (cleaned).
This is a tool: it writes its report to stdout directly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.support import (GOLDEN, GOLDEN_MODES, ROOT, clean_work, compress_file, decompress_file, engine_config,
                           mode_texts, real_config, work_dir)

WORKERS = 3
MANIFEST = GOLDEN / "manifest.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _one(mode: str, text: Path, work: Path, python_cfg: Path) -> tuple[str, bytes | str]:
    """(stem, the proven package) or (stem, why it failed)."""
    first, second, back = work / f"{text.stem}.n.hz", work / f"{text.stem}.p.hz", work / f"{text.stem}.out"
    done = compress_file(text, first, mode, "--no-fallback")
    if done.returncode != 0:
        return text.stem, f"compress exit {done.returncode}"
    done = compress_file(text, second, mode, "--no-fallback", "--config", str(python_cfg))
    if done.returncode != 0 or second.read_bytes() != first.read_bytes():
        return text.stem, "the python engine gave other bytes"
    done = decompress_file(first, back)
    if done.returncode != 0 or back.read_bytes() != text.read_bytes():
        return text.stem, "the text did not come back exactly"
    return text.stem, first.read_bytes()


def _mode(mode: str, python_cfg: Path) -> dict[str, bytes]:
    """Every golden package of ``mode``, or ``SystemExit`` naming what failed (nothing written then)."""
    work = work_dir(f"make_golden_{mode}")
    with ThreadPoolExecutor(WORKERS) as pool:
        got = dict(pool.map(lambda t: _one(mode, t, work, python_cfg), mode_texts(mode)))
    bad = {stem: why for stem, why in got.items() if isinstance(why, str)}
    if bad:
        raise SystemExit(f"{mode}: {bad}")
    return {stem: data for stem, data in got.items() if isinstance(data, bytes)}


def _text_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for mode in GOLDEN_MODES:
        for text in mode_texts(mode):
            data = text.read_bytes()
            rows[text.stem] = {"path": text.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": _sha(data),
                               "chars": len(data.decode("utf-8"))}
    return rows


def _write(mode: str, packages: dict[str, bytes]) -> dict[str, Any]:
    folder = GOLDEN / mode
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for stem, data in packages.items():
        (folder / f"{stem}.hz").write_bytes(data)
    first = next(iter(packages.values()))
    return {"id": real_config()["modes"][mode]["id"], "signature_byte": first[1], "texts": GOLDEN_MODES[mode],
            "packages": {stem: {"bytes": len(d), "sha256": _sha(d)} for stem, d in sorted(packages.items())}}


def main() -> int:
    """Make, prove and pin every mode's golden packages; write the manifest last."""
    python_cfg = engine_config("python", "make_golden_cfg")
    try:
        made = {mode: _mode(mode, python_cfg) for mode in GOLDEN_MODES}
        modes = {mode: _write(mode, packages) for mode, packages in made.items()}
    finally:
        clean_work()
    manifest = {"what": "hebzip golden set: the packages the real CLI must reproduce byte for byte",
                "python": platform.python_version(), "numpy": np.__version__, "texts": _text_rows(), "modes": modes}
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(m["packages"]) for m in modes.values())
    sys.stdout.write(f"golden: {len(modes)} modes, {total} packages, manifest written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
