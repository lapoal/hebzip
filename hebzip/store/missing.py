"""Tables that are not on this disk: one clear message saying where to get them.

Why a box of its own: the repository ships the code, not the tables - they are release downloads (and the
tables of modes ``fast`` and ``en_fast`` also live in the website folder, ``site/tables``). A fresh clone
used to stop with
"the table he_words could not be read: FileNotFoundError" (eager store) or "no mapped copy - run
convert_tables" (mmap store, which cannot be made without the tables either). Both stores now call
``require_present`` first, and a missing pinned file ends in ``TablesMissing`` (exit 3) naming the files and
the two ways to get them. The mapped store passes ``ready`` (the table is in its sealed manifest): a ready
derived copy works without the pinned files, so it is never refused for their absence. Only a file and its
folder are printed (``tables/he_words.txt``) - never a machine path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from hebzip.core.errors import TablesMissing
from hebzip.store.port import TableRef

SHOWN = 4
"""How many missing files the message names (then "and N more")."""
HOW = ("They are not part of the source repository. Get them in one of two ways:\n"
       "  1. download the tables archive (hebzip-tables-*.zip) from the project's Releases page and unzip it\n"
       "     into the project folder, or\n"
       "  2. build them from the public sources: see tables/README.md (python -m hebzip.ops.build_langpack ...).\n"
       "For modes fast and en_fast only: python tools/site/build_site.py --unpack-tables (from the website folder).\n"
       "With table_store = \"mmap\" run python -m hebzip.ops.convert_tables afterwards.")
HOW_HE = ("הטבלאות אינן חלק מקוד המקור: מורידים את קובץ הטבלאות מדף ה-Releases של הפרויקט, או בונים אותן "
          "(tables/README.md). הטבלאות של fast ושל en_fast נמצאות גם באתר: "
          "python tools/site/build_site.py --unpack-tables.")


def _shown(path: Path) -> str:
    """The file and its folder (``tables/he_words.txt``) - enough to find it, no machine path."""
    return f"{path.parent.name}/{path.name}"


def require_present(refs: Mapping[str, TableRef], names: list[str],
                    ready: Callable[[str], bool] = lambda name: False) -> None:
    """Every named table's pinned file exists, or ``TablesMissing``; a table the store holds ready (``ready``,
    e.g. in its mapped copy) needs no pinned file."""
    missing = [_shown(refs[n].path) for n in names if n in refs and not ready(n) and not refs[n].path.is_file()]
    if not missing:
        return
    more = f" and {len(missing) - SHOWN} more" if len(missing) > SHOWN else ""
    raise TablesMissing(f"hebzip: tables not found: {', '.join(missing[:SHOWN])}{more}.\n{HOW}\n{HOW_HE}")
