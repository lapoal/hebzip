"""How a pinned table is stored in the website: as-is, or gzip-compressed when that saves a lot.

Why: the English word-pair table is 42 MB as an ``.npz`` without compression but 6 MB gzipped;
static hosts do not compress binary files, so the first visit downloaded the raw size. A table whose gzip is at
least ``MIN_SAVING`` smaller is shipped as ``<path>.gz`` (level 9, no timestamp: the same table = the same
file); the browser un-gzips it with its built-in ``DecompressionStream`` and checks the sha256 of the RESULT
against the pinned row - so what the engine reads is byte for byte the pinned table. ``unpack`` does the same
for the native tool: a fresh clone gets the tables of the website's modes into ``tables/`` without a download.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

MIN_SAVING = 0.2
"""Ship gzipped when gzip is at least 20% smaller (compressed ``.npz`` files gain nothing and stay as-is)."""


def write_table(data: bytes, target: Path) -> None:
    """``target`` (raw) or ``target.gz`` - whichever the rule above picks."""
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(packed) <= (1 - MIN_SAVING) * len(data):
        Path(f"{target}.gz").write_bytes(packed)
    else:
        target.write_bytes(data)


def entry(site: Path, name: str) -> dict[str, Any]:
    """The build.json row of one fetched file: what to download, its size, the sha256 of the bytes it stands for."""
    gz = site / f"{name}.gz"
    if gz.is_file():
        raw = gzip.decompress(gz.read_bytes())
        return {"file": f"{name}.gz", "bytes": gz.stat().st_size, "raw_bytes": len(raw), "gzip": True,
                "sha256": hashlib.sha256(raw).hexdigest()}
    data = (site / name).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_table(site: Path, name: str) -> bytes:
    """The table bytes the website holds for ``name`` (un-gzipped when shipped gzipped)."""
    gz = site / f"{name}.gz"
    return gzip.decompress(gz.read_bytes()) if gz.is_file() else (site / name).read_bytes()


def unpack(site: Path, tables: dict[str, tuple[str, str]], root: Path) -> list[str]:
    """Write the website's tables into the project (``name -> (relative path, pinned sha256)``), each checked
    against its pin first; an existing file is overwritten only by the same pinned bytes. Returns the paths."""
    written = []
    for path, pinned in tables.values():
        data = read_table(site, path)
        if hashlib.sha256(data).hexdigest() != pinned:
            raise ValueError(f"{path} in the website is not the pinned table")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(path)
    return written
