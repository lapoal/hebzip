"""What a mode is, and the signature of everything that shapes its bits.

Why a signature in every package: the rows and the tables are the second half of the format. Now
that every model number is a config row, changing a row changes the bits just like changing a table
does - so the signature covers both: the canonical JSON of the mode's rows (plus the language pack)
and the sha256 of every table the mode reads. The receiver refuses a package whose header byte(s)
disagree (``TableMismatch``) instead of decoding garbage. Pure computation; zero I/O here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    """The facts about one mode the services need (copied out of the config row)."""

    name: str
    mode_id: int
    kind: str
    retired: bool


def format_signature(rows: Mapping[str, object], table_shas: Mapping[str, str]) -> bytes:
    """sha256 over the canonical JSON of the bit-shaping rows and the table sha256s -> 32 bytes."""
    canon = json.dumps({"rows": rows, "tables": dict(sorted(table_shas.items()))}, sort_keys=True,
                       ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("ascii")).digest()
