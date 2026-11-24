"""File signatures cached by (path, size, mtime_ns) - one cache file for every table check.

Why here: both table stores check files against a
sha256 before they are read - the eager store the pinned tables, the mapped store its derived copy -
and hashing ~200 MB on every start would cost more than reading. A changed size or mtime is always
re-hashed, so a changed file can never pass on an old entry. The cache is written atomically and only
when it changed; an unreadable cache means "hash everything" (a cache, never a source of truth).
Keys are paths relative to the cache's own folder: the cache never holds an absolute
path, so it says nothing about the machine and still matches when the whole project folder moves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hebzip.core.files import write_bytes_atomic
from hebzip.core.logging import get_logger
from hebzip.core.signature import file_sig

LOG = get_logger("sigcache")
SigCache = dict[str, tuple[int, int, str]]


def _load(path: Path | None) -> SigCache:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): (int(v[0]), int(v[1]), str(v[2])) for k, v in raw.items()}
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        LOG.warning("signature cache unreadable - hashing everything")
        return {}


class FileSigs:
    """The sha256 of files, through the cache at ``where`` (``None`` = always hash)."""

    def __init__(self, where: Path | None) -> None:
        self.where = where
        self.cache = _load(where)
        self.before = dict(self.cache)

    def sha(self, path: Path) -> str:
        """sha256 hex of ``path`` (``OSError`` when it cannot be read)."""
        stat = path.stat()
        key = self._key(path)
        known = self.cache.get(key)
        if known is not None and known[0] == stat.st_size and known[1] == stat.st_mtime_ns:
            return known[2]
        sha = file_sig(path)
        self.cache[key] = (stat.st_size, stat.st_mtime_ns, sha)
        return sha

    def _key(self, path: Path) -> str:
        """``path`` relative to the cache's folder (posix form); absolute only if no relative form exists."""
        if self.where is None:
            return str(path)
        try:
            return Path(os.path.relpath(path.resolve(), self.where.resolve().parent)).as_posix()
        except ValueError:  # another drive on Windows: there is no relative path
            return str(path)

    def save(self) -> None:
        """Write the cache back when something new was hashed."""
        if self.where is not None and self.cache != self.before:
            write_bytes_atomic(self.where, json.dumps({k: list(v) for k, v in self.cache.items()},
                                                      sort_keys=True).encode())
            self.before = dict(self.cache)
