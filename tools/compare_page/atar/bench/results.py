"""The results file: numbers and names only, rewritten after every text (a crash loses one text at most).

Why after every text: a full run takes over an hour on a small laptop, so a
measurement is saved as it goes and can be resumed. The write is atomic (a temp file in the same folder,
then replace), so an interrupted write never leaves half a file. No text and no file content is ever
written here - the machine-and-material rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from atar.core.errors import ConfigError


def load(path: Path) -> dict[str, Any]:
    """The results so far (empty skeleton when there are none)."""
    if not path.is_file():
        return {"tools": {}, "rows": []}
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"לא הצלחתי לקרוא את קובץ התוצאות: {type(error).__name__}") from error
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    """Write atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(temp, path)


def done_pairs(data: dict[str, Any]) -> set[tuple[str, str]]:
    """(tool, text) pairs already measured successfully - skipped on a resumed run."""
    return {(r["tool"], r["text"]) for r in data["rows"] if r["exact"]}


def put(data: dict[str, Any], row: dict[str, Any]) -> None:
    """Add a measured pair, replacing an earlier failed try of the same pair."""
    data["rows"] = [r for r in data["rows"] if (r["tool"], r["text"]) != (row["tool"], row["text"])]
    data["rows"].append(row)
