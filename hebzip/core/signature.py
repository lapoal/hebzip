"""Content signatures - sha256 of what a thing *is*, never of when it was made (the cache-by-content-signature rule).

One module so the golden net, the config table
check and the package header all hash the same way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def file_sig(path: Path, chunk: int = CHUNK) -> str:
    """sha256 hex of a file's bytes, read in 1 MB chunks (the tables are tens of MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def short(sig: str, n: int = 8) -> str:
    """The first ``n`` hex characters - what the log may show (never content)."""
    return sig[:n]
