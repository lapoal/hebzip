"""The manifest of the derived table copy: what each file is, and which pinned table it came from.

Why: the derived copy is not pinned by the config - the config pins the SOURCE tables. The manifest ties
the two: per table the pinned source's sha256 (must equal the config row, or the copy was made from
another table and is refused), and per derived file its sha256, dtype, shape and a hash of its VALUES
(the same for the source array and its narrowed copy - proven by the converter). The manifest seals
itself (``self_sha256`` over its canonical JSON), so a hand-edited or cut manifest is refused too.
Layout: ``<dir>/manifest.json``; ``<dir>/<table>/<array>.npy`` per array of an ``.npz`` table,
``<dir>/<table>/array.npy`` for an ``.npy`` table, ``<dir>/<table>/<file name>`` for a text/JSON table
(copied byte for byte). The manifest is written last, so a cut conversion leaves no valid copy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch
from hebzip.core.files import write_bytes_atomic

MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
SEAL = "self_sha256"
CANONICAL_INT = {"i": "<i8", "u": "<u8"}
"""Integer kind -> the width every integer array is hashed in (so a narrowed copy hashes like its source)."""


@dataclass(frozen=True)
class FileEntry:
    """One derived file. ``dtype``/``shape``/``values_sha256`` are ``None`` for a copied text/JSON file."""

    file: str
    sha256: str
    bytes: int
    dtype: str | None
    source_dtype: str | None
    shape: list[int] | None
    values_sha256: str | None


@dataclass(frozen=True)
class TableEntry:
    """One table: its pinned source and its derived files (key = array name, ``""`` = the only one)."""

    source_sha256: str
    source_bytes: int
    files: dict[str, FileEntry]


def values_sha256(array: npt.NDArray[np.generic]) -> str:
    """sha256 of an array's shape, signedness and values - not of its width or byte order."""
    kind = array.dtype.kind
    wide = CANONICAL_INT.get(kind, array.dtype.newbyteorder("<").str)
    digest = hashlib.sha256(f"{kind}|{wide}|{list(array.shape)}|".encode())
    digest.update(np.ascontiguousarray(array, dtype=wide).tobytes())
    return digest.hexdigest()


def seal(raw: dict[str, Any]) -> str:
    """The manifest's own sha256: canonical JSON of everything but the seal."""
    body = {k: v for k, v in raw.items() if k != SEAL}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def write_manifest(folder: Path, tables: dict[str, TableEntry]) -> None:
    """Write the sealed manifest (atomic)."""
    raw: dict[str, Any] = {"format": MANIFEST_FORMAT, "tables": {n: asdict(e) for n, e in sorted(tables.items())}}
    raw[SEAL] = seal(raw)
    write_bytes_atomic(folder / MANIFEST, json.dumps(raw, sort_keys=True, indent=1).encode("utf-8"))


def _entry(raw: dict[str, Any]) -> TableEntry:
    files = {key: FileEntry(**value) for key, value in raw["files"].items()}
    return TableEntry(str(raw["source_sha256"]), int(raw["source_bytes"]), files)


def read_manifest(folder: Path) -> dict[str, TableEntry]:
    """The manifest's tables; missing, unreadable, unsealed or of another format -> ``TableMismatch``."""
    path = folder / MANIFEST
    if not path.is_file():
        raise TableMismatch(f"אין עותק-טבלאות ממופה ב-{folder.name} - הרץ python -m hebzip.ops.convert_tables")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get(SEAL) != seal(raw) or raw.get("format") != MANIFEST_FORMAT:
            raise TableMismatch(f"קובץ {MANIFEST} נערך או פגום (החותם שלו אינו תואם) - הרץ את הממיר מחדש")
        return {str(name): _entry(entry) for name, entry in raw["tables"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise TableMismatch(f"קובץ {MANIFEST} לא נקרא: {type(error).__name__} - הרץ את הממיר מחדש") from error


def merged(folder: Path, new: dict[str, TableEntry]) -> dict[str, TableEntry]:
    """An existing valid manifest's other tables plus ``new`` (a partial conversion keeps the rest)."""
    try:
        old = read_manifest(folder)
    except TableMismatch:
        old = {}
    return {**old, **new}
