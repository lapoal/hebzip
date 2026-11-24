"""The material folder: the only place uploaded text may exist, and only while a job runs.

the machine-and-material rule (code is machine, text is material): the upload's text, every package and
every decompressed copy live in ``<material>/job-<n>/`` and that folder is deleted when the
job ends - also on error. Leftovers from a crash or a closed window are wiped at server
start. Wiping touches only children of the declared folder, never anything else.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from atar.core.logging import get_logger

_log = get_logger("material")


def wipe_all(material: Path) -> int:
    """Delete everything inside the material folder; return how many entries were removed."""
    material.mkdir(parents=True, exist_ok=True)
    removed = 0
    for child in list(material.iterdir()):
        remove(child)
        removed += 1
    if removed:
        _log.warning("leftovers wiped at start", entries=removed)
    return removed


def scratch(material: Path, job: int) -> Path:
    """A fresh, empty folder for one job."""
    folder = material / f"job-{job}"
    remove(folder)
    folder.mkdir(parents=True)
    return folder


def remove(path: Path) -> None:
    """Delete a file or folder tree inside the material folder (read-only bits cleared)."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=_force)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _force(func: object, target: str, error: BaseException) -> None:
    """rmtree helper: clear a read-only flag and retry once (Windows)."""
    del func, error
    p = Path(target)
    p.chmod(0o666)
    if p.is_dir():
        p.rmdir()
    else:
        p.unlink()


def is_empty(material: Path) -> bool:
    """True when no job folder or stray file is left."""
    return not material.exists() or not any(material.iterdir())
