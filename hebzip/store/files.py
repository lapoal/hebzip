"""Readers for the table formats: a word list, an ``.npz`` of arrays, an ``.npy`` (read or mapped), JSON.

Why a layer of its own: ``domain`` has zero I/O, so something must turn files into plain arrays and
lists. These readers decide nothing about the format's meaning - the sha256 check already ran
(``config/tables.py``); a file that still cannot be read is a ``TableMismatch`` naming the file.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch

Arrays = dict[str, npt.NDArray[np.generic]]


def read_words(path: Path) -> list[str]:
    """One word per line (UTF-8, ``\n``); the last line ends with ``\n``."""
    try:
        return path.read_text(encoding="utf-8").split("\n")[:-1]
    except (OSError, UnicodeDecodeError) as error:
        raise TableMismatch(f"רשימת המילים {path.name} לא נקראה: {error}") from error


def read_arrays(path: Path) -> Arrays:
    """Every array of an ``.npz`` file (no pickles)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise TableMismatch(f"הטבלה {path.name} לא נקראה: {error}") from error


def read_matrix(path: Path, key: str) -> npt.NDArray[np.int64]:
    """An integer matrix stored as ``{key: [[...], ...]}`` in JSON."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return np.array(raw[key], dtype=np.int64)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TableMismatch(f"הטבלה {path.name} לא נקראה: {error}") from error


def read_array(path: Path) -> npt.NDArray[np.generic]:
    """One ``.npy`` array (no pickles)."""
    try:
        array: npt.NDArray[np.generic] = np.load(path, allow_pickle=False)
        return array
    except (OSError, ValueError) as error:
        raise TableMismatch(f"הטבלה {path.name} לא נקראה: {error}") from error


def read_mapped(path: Path) -> npt.NDArray[np.generic]:
    """One ``.npy`` array as a read-only memory map (the derived table copy) - no pickles;
    the operating system reads a page only when it is touched."""
    try:
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        # a plain read-only view of the map: np.memmap's subclass hooks cost ~5x per slice
        array: npt.NDArray[np.generic] = np.asarray(mapped)
        return array
    except (OSError, ValueError) as error:
        raise TableMismatch(f"הטבלה {path.name} לא נפתחה: {type(error).__name__}") from error


def read_weight_sets(path: Path, key: str) -> dict[tuple[int, int], list[float]]:
    """JSON ``{key: {"d,c": [floats], ...}}`` -> {(d, c): [floats]} (the speller's start weight sets)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))[key]
        out: dict[tuple[int, int], list[float]] = {}
        for name, weights in raw.items():
            a, b = (int(v) for v in str(name).split(","))
            out[(a, b)] = [float(w) for w in weights]
        return out
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise TableMismatch(f"הטבלה {path.name} לא נקראה: {error}") from error
