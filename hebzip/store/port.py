"""The table-store socket: where a mode's tables come from.

Why a socket (the socket rule (contract + real drawer + plate) and the change-without-code rule - "a method
someone may want otherwise = a word"): the pinned
tables are compressed ``.npz`` / ``.npy`` / text files, read whole at every start. A derived copy of
raw, narrowed ``.npy`` files can instead be mapped, so the operating system reads only the pages a
message touches. Which one a run uses is the word ``[hebzip].table_store``:
  ``eager``  the pinned files, read whole (``store/eager.py``) - exactly the old behaviour;
  ``mmap``   the derived copy + its manifest (``store/mapped.py``, made by ``ops/convert_tables.py``);
  ``null``   the plate (below): no tables at all - a mode that needs one is refused, never fed
             invented values; a mode without tables (``generic``) still works.
Every store answers ``verify`` with the PINNED sha256 of each table (the config rows), so the format
signature - and every package byte - is the same whichever store read the tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from hebzip.core.errors import TableMismatch

Arrays = dict[str, npt.NDArray[np.generic]]
WeightSets = dict[tuple[int, int], list[float]]


@dataclass(frozen=True)
class TableRef:
    """One ``[tables.<name>]`` as a store needs it: the pinned file and its pinned sha256."""

    path: Path
    sha256: str


@runtime_checkable
class TableStorePort(Protocol):
    """Where the tables of a run come from (checked first, then read)."""

    name: str

    def verify(self, names: list[str]) -> dict[str, str]:
        """Check the named tables -> {name: pinned sha256}; ``TableMismatch`` on any doubt."""
        ...

    def arrays(self, name: str) -> Arrays:
        """Every array of an ``.npz`` table."""
        ...

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """The array of an ``.npy`` table."""
        ...

    def words(self, name: str) -> list[str]:
        """A word-list table."""
        ...

    def matrix(self, name: str, key: str) -> npt.NDArray[np.int64]:
        """An integer matrix stored as JSON ``{key: [[...]]}``."""
        ...

    def weight_sets(self, name: str, key: str) -> WeightSets:
        """JSON weight sets ``{key: {"d,c": [floats]}}``."""
        ...


class NullTableStore:
    """The plate: holds no table. Verifying nothing is fine; asking for any table is ``TableMismatch``."""

    name = "null"

    def verify(self, names: list[str]) -> dict[str, str]:
        """Only the empty list passes."""
        if names:
            raise _refused(names[0])
        return {}

    def arrays(self, name: str) -> Arrays:
        """Refused."""
        raise _refused(name)

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """Refused."""
        raise _refused(name)

    def words(self, name: str) -> list[str]:
        """Refused."""
        raise _refused(name)

    def matrix(self, name: str, key: str) -> npt.NDArray[np.int64]:
        """Refused."""
        raise _refused(name)

    def weight_sets(self, name: str, key: str) -> WeightSets:
        """Refused."""
        raise _refused(name)


def _refused(name: str) -> TableMismatch:
    return TableMismatch(f"הטבלה {name}: אין מאגר-טבלאות ([hebzip].table_store = \"null\")")
