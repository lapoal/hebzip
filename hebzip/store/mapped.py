"""The mapped table store: the derived copy of raw narrowed ``.npy`` files, opened as memory maps.

Why: a map costs nothing until a page is touched - the operating system reads only the parts of a table
a message uses, and they stay shared in its file cache between runs instead of being inflated and copied
into every process. Nothing is trusted on its word (the ready-only-after-proof rule): ``verify`` refuses
(``TableMismatch``, exit 3) unless the manifest is sealed, names each table with the SAME pinned sha256
as the config row, and every derived file has the sha256 the manifest records (through the shared
signature cache, so a start does not re-hash 200 MB); each map is opened with the dtype and shape the
manifest records. It answers with the pinned sha256, so the format signature is the eager store's.
Windows: a mapped file cannot be replaced while a process maps it - convert with no run in progress.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch
from hebzip.core.sigcache import FileSigs
from hebzip.core.signature import short
from hebzip.store.files import read_mapped, read_matrix, read_weight_sets, read_words
from hebzip.store.manifest import MANIFEST, FileEntry, TableEntry, read_manifest
from hebzip.store.missing import require_present
from hebzip.store.port import Arrays, TableRef, WeightSets

ONLY = ""
"""The key of the single file of an ``.npy`` or text/JSON table."""


class MmapTableStore:
    """The derived copy at ``folder``, checked against its manifest and the pinned rows, then mapped."""

    name = "mmap"

    def __init__(self, refs: Mapping[str, TableRef], folder: Path, sig_cache: Path | None) -> None:
        self.refs = dict(refs)
        self.folder = folder
        self.sig_cache = sig_cache
        self._tables: dict[str, TableEntry] | None = None

    def _entry(self, name: str) -> TableEntry:
        if self._tables is None:
            self._tables = read_manifest(self.folder)
        entry = self._tables.get(name)
        if entry is None:
            raise TableMismatch(f"הטבלה {name} אינה בעותק הממופה - הרץ python -m hebzip.ops.convert_tables")
        return entry

    def verify(self, names: list[str]) -> dict[str, str]:
        """Manifest sealed, made from the pinned table, every derived file intact -> {name: pinned sha256}.
        No derived copy and no pinned files either (a fresh clone) -> ``TablesMissing`` saying where to get them."""
        require_present(self.refs, names, self._ready)
        sigs = FileSigs(self.sig_cache)
        out: dict[str, str] = {}
        try:
            for name in names:
                entry, ref = self._entry(name), self.refs.get(name)
                if ref is None or entry.source_sha256 != ref.sha256:
                    raise TableMismatch(f"העותק הממופה של {name} נעשה מטבלה אחרת (sha256 "
                                        f"{short(entry.source_sha256)}) - הרץ את הממיר מחדש")
                for item in entry.files.values():
                    self._check(name, item, sigs)
                out[name] = ref.sha256
        finally:
            sigs.save()
        return out

    def _ready(self, name: str) -> bool:
        """Is ``name`` in the derived copy's manifest? (no manifest = nothing is ready)"""
        if not (self.folder / MANIFEST).is_file():
            return False
        try:
            self._entry(name)
        except TableMismatch:
            return False
        return True

    def _check(self, name: str, item: FileEntry, sigs: FileSigs) -> None:
        try:
            sha = sigs.sha(self.folder / item.file)
        except OSError as error:
            raise TableMismatch(f"קובץ {item.file} של {name} חסר בעותק הממופה ({type(error).__name__})") from error
        if sha != item.sha256:
            raise TableMismatch(f"קובץ {item.file} של {name} שונה מהרשום במניפסט (sha256 {short(sha)})")

    def _open(self, name: str, item: FileEntry) -> npt.NDArray[np.generic]:
        array = read_mapped(self.folder / item.file)
        if array.dtype.str != item.dtype or list(array.shape) != item.shape:
            raise TableMismatch(f"קובץ {item.file} של {name} אינו בצורה שבמניפסט ({array.dtype.str}, {array.shape})")
        return array

    def arrays(self, name: str) -> Arrays:
        """Every array of an ``.npz`` table, mapped."""
        entry = self._entry(name)
        return {key: self._open(name, item) for key, item in entry.files.items() if item.dtype is not None}

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """The array of an ``.npy`` table, mapped."""
        return self._open(name, self._only(name))

    def _only(self, name: str) -> FileEntry:
        item = self._entry(name).files.get(ONLY)
        if item is None:
            raise TableMismatch(f"הטבלה {name} אינה קובץ יחיד בעותק הממופה")
        return item

    def words(self, name: str) -> list[str]:
        """The word list (a byte-for-byte copy of the pinned file)."""
        return read_words(self.folder / self._only(name).file)

    def matrix(self, name: str, key: str) -> npt.NDArray[np.int64]:
        """The JSON matrix (a copy of the pinned file)."""
        return read_matrix(self.folder / self._only(name).file, key)

    def weight_sets(self, name: str, key: str) -> WeightSets:
        """The JSON weight sets (a copy of the pinned file)."""
        return read_weight_sets(self.folder / self._only(name).file, key)
