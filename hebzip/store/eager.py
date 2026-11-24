"""The eager table store: the pinned files, checked against their sha256 rows and read whole.

Why: this is exactly what every run did before the store became a socket - the sha256 check moved here
from ``config/tables.verify_tables`` (same messages, same cache), the readers are ``store/files.py``. The
word ``eager`` keeps the old behaviour byte for byte; it is also the store the table builder reads with
(``ops/build.py``: a rebuild must read the pinned files, never a derived copy that may be stale).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch
from hebzip.core.sigcache import FileSigs
from hebzip.core.signature import short
from hebzip.store.files import read_array, read_arrays, read_matrix, read_weight_sets, read_words
from hebzip.store.missing import require_present
from hebzip.store.port import Arrays, TableRef, WeightSets


class EagerTableStore:
    """The pinned table files, read whole."""

    name = "eager"

    def __init__(self, refs: Mapping[str, TableRef], sig_cache: Path | None) -> None:
        self.refs = dict(refs)
        self.sig_cache = sig_cache

    def path(self, name: str) -> Path:
        """The pinned file of ``[tables.<name>]``."""
        ref = self.refs.get(name)
        if ref is None:
            raise TableMismatch(f"הטבלה {name} אינה בהגדרות")
        return ref.path

    def verify(self, names: list[str]) -> dict[str, str]:
        """Each named pinned file against its sha256 row -> {name: sha256} (absent files: ``TablesMissing``)."""
        require_present(self.refs, names)
        sigs = FileSigs(self.sig_cache)
        out: dict[str, str] = {}
        try:
            for name in names:
                try:
                    sha = sigs.sha(self.path(name))
                except OSError as error:
                    raise TableMismatch(f"הטבלה {name} לא נקראה: {type(error).__name__}") from error
                pinned = self.refs[name].sha256
                if sha != pinned:
                    raise TableMismatch(f"הטבלה {name} שונה מההגדרות (sha256 {short(sha)} במקום {short(pinned)})")
                out[name] = sha
        finally:
            sigs.save()
        return out

    def arrays(self, name: str) -> Arrays:
        """Every array of the ``.npz`` file."""
        return read_arrays(self.path(name))

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """The ``.npy`` file's array."""
        return read_array(self.path(name))

    def words(self, name: str) -> list[str]:
        """The word list."""
        return read_words(self.path(name))

    def matrix(self, name: str, key: str) -> npt.NDArray[np.int64]:
        """The JSON matrix."""
        return read_matrix(self.path(name), key)

    def weight_sets(self, name: str, key: str) -> WeightSets:
        """The JSON weight sets."""
        return read_weight_sets(self.path(name), key)
