"""Safe file access: atomic write and size-limited read.

``write_bytes_atomic`` writes a writer-private side file
named by pid+thread, then ``os.replace`` through ``locked_file``. Why: an output must never be
half-written - a cut package would look like a corrupt one - and two processes may write at once.
``read_bytes_limited`` refuses a file bigger than the configured cap *before* reading it all.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from hebzip.core.errors import InputError
from hebzip.core.logging import get_logger
from hebzip.core.retry import locked_file

LOG = get_logger("files")


def _temporary(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """All or nothing: a side file, then replace - never a half-written ``path``."""
    temporary = _temporary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_bytes(data)
        locked_file(lambda: os.replace(temporary, path))
    except OSError:
        _discard(temporary)
        raise


def read_bytes_limited(path: Path, max_bytes: int) -> bytes:
    """The file's bytes; missing or bigger than ``max_bytes`` -> ``InputError`` (Hebrew)."""
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise InputError(f"הקובץ לא נמצא: {path}") from error
    except OSError as error:
        raise InputError(f"הקובץ לא נקרא: {path} ({type(error).__name__})") from error
    if size > max_bytes:
        raise InputError(f"הקובץ גדול מדי: {size} בתים (המותר {max_bytes})")
    try:
        return locked_file(path.read_bytes)
    except OSError as error:
        raise InputError(f"הקובץ לא נקרא: {path} ({type(error).__name__})") from error


def _discard(temporary: Path) -> None:
    try:
        if temporary.is_file():
            temporary.unlink()
    except OSError as error:
        LOG.warning("side file not removed", error=type(error).__name__)
