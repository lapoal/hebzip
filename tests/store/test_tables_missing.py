"""A fresh clone without the tables: one clear message saying where to get them (``store/missing.py``).

Why (the public release): the tables are release downloads, not part of the repository. Before,
a clone stopped with "the table he_words could not be read: FileNotFoundError" (eager) or "no mapped copy -
run convert_tables" (mmap - which cannot be made without the tables either). Through the real CLI, in both
stores: exit 3, the missing files named, both ways to get them, no machine path. A ready mapped copy still
works without the pinned files (the copy is enough), and a store with its pinned files but no mapped copy
keeps the converter's own message.
"""

from __future__ import annotations

from pathlib import Path

from tests.support import ROOT, TEXTS, edit_config, ready_store_config, run_cli, store_config, work_dir

EXIT_TABLES = 3
MOVED = ('path = "tables/he_words.txt"', 'path = "tables/not_here/he_words.txt"')
"""The word list of every Hebrew mode, pointed at a file that does not exist (a clone without tables)."""


def _compress(config: Path, name: str) -> tuple[int, str]:
    work = work_dir(name)
    done = run_cli("compress", "--config", config, "--mode", "fast", "--force", TEXTS / "own_message.txt",
                   work / "x.hz")
    return done.returncode, done.stderr.decode("utf-8", "replace")


def _says_where(err: str) -> None:
    assert "tables not found: not_here/he_words.txt" in err
    assert "Releases" in err and "tables/README.md" in err and "--unpack-tables" in err
    assert str(ROOT) not in err and str(ROOT.parent) not in err


def test_eager_store_without_tables_says_where_to_get_them() -> None:
    code, err = _compress(store_config("missing_eager", "eager", MOVED), "missing_eager_run")
    assert code == EXIT_TABLES
    _says_where(err)


def test_mmap_store_without_copy_or_tables_says_where_to_get_them() -> None:
    code, err = _compress(store_config("missing_mmap", "mmap", MOVED), "missing_mmap_run")
    assert code == EXIT_TABLES
    _says_where(err)


def test_mmap_store_with_tables_but_no_copy_keeps_the_converter_message() -> None:
    code, err = _compress(store_config("missing_copy", "mmap"), "missing_copy_run")
    assert code == EXIT_TABLES
    assert "convert_tables" in err and "tables not found" not in err


def test_a_ready_mapped_copy_needs_no_pinned_files() -> None:
    config = ready_store_config("missing_ready", "mmap", ("fast",))
    edit_config(config, MOVED)
    code, err = _compress(config, "missing_ready_run")
    assert code == 0, err[-400:]


def test_a_mapped_copy_of_other_modes_does_not_hide_missing_tables() -> None:
    """A clone that unpacked and converted only fast's tables, then asks for max: name max's missing files."""
    config = ready_store_config("missing_partial", "mmap", ("fast",))
    edit_config(config, ('path = "tables/corpus_gap_starts.npz"', 'path = "tables/not_here/corpus_gap_starts.npz"'))
    work = work_dir("missing_partial_run")
    done = run_cli("compress", "--config", config, "--mode", "max", "--force", TEXTS / "own_message.txt",
                   work / "x.hz")
    assert done.returncode == EXIT_TABLES
    assert "tables not found: not_here/corpus_gap_starts.npz" in done.stderr.decode("utf-8", "replace")
