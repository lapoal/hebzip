"""The signature cache never writes an absolute path.

Why: the cache used to key every file by its absolute path, so a cache file said where the project lived on
this machine (and stopped matching when the folder moved). Keys are now relative to the cache's own folder.
* after hashing a table the cache file holds only the relative key (``../tables/x.npy``), no drive, no root;
* a second ``FileSigs`` reads the entry back under the same key (the cache still works);
* a changed file is hashed again (a stale entry can never pass).
"""

from __future__ import annotations

import json
from pathlib import Path

from hebzip.core.sigcache import FileSigs
from hebzip.core.signature import file_sig


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    table = tmp_path / "tables" / "x.npy"
    table.parent.mkdir()
    table.write_bytes(b"some table bytes")
    return table, tmp_path / ".cache" / "sigs.json"


def test_the_cache_holds_relative_keys_only(tmp_path: Path) -> None:
    table, where = _setup(tmp_path)
    sigs = FileSigs(where)
    assert sigs.sha(table) == file_sig(table)
    sigs.save()
    raw = where.read_text(encoding="utf-8")
    assert list(json.loads(raw)) == ["../tables/x.npy"]
    assert str(tmp_path) not in raw and str(tmp_path.resolve()) not in raw and ":" not in next(iter(json.loads(raw)))


def test_a_second_reader_finds_the_entry_and_a_change_is_hashed_again(tmp_path: Path) -> None:
    table, where = _setup(tmp_path)
    first = FileSigs(where)
    first.sha(table)
    first.save()
    second = FileSigs(where)
    assert set(second.cache) == {"../tables/x.npy"} and second.sha(table) == file_sig(table)
    table.write_bytes(b"other bytes, other size")
    assert second.sha(table) == file_sig(table) != first.cache["../tables/x.npy"][2]
