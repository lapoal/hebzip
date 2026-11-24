"""Writers for the table formats the readers in ``files.py`` read back (used by the ops table builder).

Why here (``store``): the one layer that touches table files. Why byte-for-byte deterministic: a
table is pinned by its sha256 in the config, so building the same table twice from the same corpus
must give the same file - ``np.savez`` stamps the current time into its zip entries, so the archive
is written here with a fixed entry time instead. Every write is atomic (``core.files``).
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hebzip.core.files import write_bytes_atomic

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
"""The earliest time a zip entry can carry - a fixed stamp, so equal arrays give equal files."""


def _npy(array: npt.NDArray[np.generic]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_words(path: Path, words: list[str]) -> None:
    """One word per line, UTF-8, every line ended by ``\\n`` (what ``read_words`` reads)."""
    write_bytes_atomic(path, "".join(w + "\n" for w in words).encode("utf-8"))


def write_arrays(path: Path, arrays: Mapping[str, npt.NDArray[np.generic]]) -> None:
    """An ``.npz`` (no pickles), entries in name order with a fixed time stamp."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_EPOCH), _npy(arrays[name]))
    write_bytes_atomic(path, buffer.getvalue())


def write_array(path: Path, array: npt.NDArray[np.generic]) -> None:
    """One ``.npy`` array."""
    write_bytes_atomic(path, _npy(array))


def write_json(path: Path, value: object) -> None:
    """Canonical JSON (sorted keys, Python's default separators) - the matrix and weight-set tables.

    Why the default separators: the research prototype wrote its weight tables with ``json.dumps``; the
    same format lets a rebuilt Hebrew weights table come out byte-identical to the pinned one."""
    write_bytes_atomic(path, json.dumps(value, sort_keys=True).encode("utf-8"))


def copy_file(source: Path, target: Path) -> None:
    """A byte-for-byte copy (the text/JSON tables of the derived copy)."""
    write_bytes_atomic(target, source.read_bytes())
