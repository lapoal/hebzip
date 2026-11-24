"""Build the tables of one language's modes from one corpus file (the change-without-code rule: "another country").

    python -m hebzip.ops.build_langpack --language latin_toy --corpus langpacks/latin_toy/corpus.txt
    python -m hebzip.ops.build_langpack --language he --corpus <hewiki.jsonl> --corpus-format jsonl
                                        --tables template_m128,prefix_prior --out <folder>

Why: a new language must be rows + tables, and the tables must be buildable without the research prototype.
Every table with a ``[tables.<name>.recipe]`` that a mode of the language reads is built (``--tables``
picks some); a table without a recipe was given and is only read (the Hebrew word list). Each is
written to its ``[tables.<name>].path`` - or into ``--out`` (same file names) to compare a rebuild
with the pinned files without touching them - and the tool prints the ``sha256`` / ``bytes`` rows to
paste into the config. The config is only shape-checked plus its recipe rows (its tables may not
exist yet - building them is the point). This is a tool: it writes its report to stdout directly.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hebzip.config.load import DEFAULT_CONFIG, PROJECT_ROOT, read_toml
from hebzip.config.schema import Settings
from hebzip.config.validate import resolve
from hebzip.config.validate_recipe import check_recipe
from hebzip.core.errors import ConfigError, HebzipError
from hebzip.core.files import read_bytes_limited
from hebzip.core.shapes import ShapeError, build
from hebzip.core.signature import file_sig
from hebzip.ops.build import Build
from hebzip.ops.corpus import READERS
from hebzip.ops.recipes import RECIPES

CORPUS_MAX_BYTES = 1 << 30
"""Refuse a corpus file above 1 GB before reading it (the tool keeps the whole corpus in memory)."""
NEXT_STEP = ("# next: paste these rows into the config; with [hebzip].table_store = \"mmap\" then run "
             "python -m hebzip.ops.convert_tables (the mapped copy does not hold the new tables until then)")
"""The last line of every report (a TOML comment, harmless if pasted). Why always, with no store check: with
the real config on ``mmap`` a freshly built table is refused at the next
compress ("not in the mapped copy") until the converter runs - the builder is where the user learns the
order; asking "which store am I" here would be a mode question inside a tool."""


def _shaped(path: Path) -> Settings:
    try:
        settings = build(Settings, read_toml(path))
    except ShapeError as error:
        raise ConfigError(f"{path.name}: {'; '.join(p.render() for p in error.problems)}") from error
    for name, table in settings.tables.items():
        if table.recipe is not None:
            check_recipe(f"[tables.{name}.recipe]", table.recipe, path.name, frozenset(settings.tables))
    return settings


def targets(b: Build, wanted: list[str] | None) -> list[str]:
    """The tables to build: the language's tables that have a recipe (or the ``wanted`` ones, each checked)."""
    buildable = sorted(n for n in b.readers if b.recipe(n) is not None)
    if wanted is None:
        return buildable
    for name in wanted:
        if name not in b.readers:
            raise ConfigError(f"[tables.{name}]: no mode of [languages.{b.language}] reads it")
        if b.recipe(name) is None:
            raise ConfigError(f"[tables.{name}]: no [tables.{name}.recipe] - this table is given, not built")
    return wanted


def run(config: Path, language: str, corpus: Path, form: str, wanted: list[str] | None,
        out: Path | None, vectors: Path | None = None) -> list[str]:
    """Build and write the tables; returns the config rows to paste (one ``[tables.*]`` block each)."""
    settings = _shaped(config)
    text = read_bytes_limited(corpus, CORPUS_MAX_BYTES).decode("utf-8")
    b = Build(settings, language, READERS[form](text), PROJECT_ROOT, RECIPES, vectors)
    rows: list[str] = []
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for table in targets(b, wanted):
        written = settings.tables[table].path
        path = (out / Path(written).name) if out is not None else resolve(PROJECT_ROOT, written)
        b.make(table).write(path)
        shown = path.as_posix() if out is not None else written
        rows += [f"[tables.{table}]", f'path = "{shown}"', f'sha256 = "{file_sig(path)}"',
                 f"bytes = {path.stat().st_size}", ""]
    return rows + [NEXT_STEP]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: 0 = every table written; the error's exit code otherwise."""
    parser = argparse.ArgumentParser(prog="hebzip.ops.build_langpack")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--language", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-format", choices=sorted(READERS), default="lines")
    parser.add_argument("--tables", help="comma-separated table names (default: every table with a recipe)")
    parser.add_argument("--out", type=Path, help="write into this folder instead of the configured paths")
    parser.add_argument("--vectors", type=Path, help="word vectors .npy (one row per word id) for word_classes")
    args = parser.parse_args(argv)
    wanted = [t for t in args.tables.split(",") if t] if args.tables else None
    try:
        rows = run(args.config, args.language, args.corpus, args.corpus_format, wanted, args.out, args.vectors)
    except (HebzipError, UnicodeDecodeError) as error:
        sys.stderr.write(f"{error}\n")
        return error.exit_code if isinstance(error, HebzipError) else 2
    sys.stdout.write("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
