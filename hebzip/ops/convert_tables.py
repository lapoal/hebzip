"""Make the derived (mapped) copy of the pinned tables: raw narrowed ``.npy`` files + a sealed manifest.

    python -m hebzip.ops.convert_tables                       # every table of the config
    python -m hebzip.ops.convert_tables --modes fast,max      # only the tables these modes read
    python -m hebzip.ops.convert_tables --out <folder>        # somewhere else than [hebzip].table_store_dir

Why derived and never built directly: the config pins the SOURCE files by sha256 and the
format signature is made of those rows; a table built straight into the new format would have no pinned
source to be checked against. So the copy is always made from the pinned files, after their sha256 check,
and it is a cache of them in another shape - the manifest says which source each file came from.
Order (nothing half-done): (1) check the pinned files; (2) narrow EVERY array in memory - a value that
fits no allowed width (``[hebzip].narrow_bits``) stops here with ``NarrowingOverflow`` (exit 3) before one
byte is written; (3) write each file, map it back and prove it equal to the source (values and their
hash); (4) write the manifest last. Text/JSON tables are copied byte for byte (same sha256 as the pin).
The pinned tables are only read. This is a tool: it writes its report to stdout directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hebzip.config.load import DEFAULT_CONFIG, load_config, project_root
from hebzip.config.schema import Settings
from hebzip.config.tables import mode_tables
from hebzip.config.validate import resolve
from hebzip.core.errors import HebzipError, NarrowingOverflow, TableMismatch
from hebzip.core.signature import file_sig
from hebzip.services.select import table_refs
from hebzip.store.eager import EagerTableStore
from hebzip.store.files import read_mapped
from hebzip.store.manifest import FileEntry, TableEntry, merged, values_sha256, write_manifest
from hebzip.store.mapped import ONLY
from hebzip.store.narrow import narrow
from hebzip.store.write import copy_file, write_array

Array = npt.NDArray[np.generic]
Pairs = dict[str, tuple[Array, Array]]
"""Array key -> (pinned array, narrowed array)."""
LAYOUTS: dict[str, Callable[[EagerTableStore, str], dict[str, Array]]] = {
    ".npz": lambda store, name: store.arrays(name),
    ".npy": lambda store, name: {ONLY: store.array(name)},
}
"""Pinned file suffix -> its arrays. Any other suffix is a text/JSON table: copied byte for byte."""
ONLY_FILE = "array"


@dataclasses.dataclass(frozen=True)
class Planned:
    """One table ready to write: its narrowed arrays, or ``None`` = copy the file."""

    name: str
    pairs: Pairs | None


def _narrowed(name: str, arrays: dict[str, Array], bits: Sequence[int]) -> Pairs:
    out: Pairs = {}
    for key, source in arrays.items():
        try:
            out[key] = (source, narrow(source, bits))
        except NarrowingOverflow as error:
            raise NarrowingOverflow(f"הטבלה {name}, המערך {key or ONLY_FILE}: {error}") from error
    return out


def plan(store: EagerTableStore, names: list[str], bits: Sequence[int]) -> list[Planned]:
    """Every table read and narrowed in memory - an overflow refuses before anything is written."""
    planned = []
    for name in names:
        read = LAYOUTS.get(store.path(name).suffix)
        planned.append(Planned(name, None if read is None else _narrowed(name, read(store, name), bits)))
    return planned


def _write_array(out: Path, name: str, key: str, source: Array, narrowed: Array) -> FileEntry:
    rel = f"{name}/{key or ONLY_FILE}.npy"
    write_array(out / rel, narrowed)
    mapped = read_mapped(out / rel)
    same = mapped.shape == source.shape and bool(np.array_equal(mapped, source))
    values = values_sha256(mapped)
    del mapped  # Windows: release the map before anything may replace the file
    if not same or values != values_sha256(source):
        raise TableMismatch(f"הטבלה {name}, המערך {key}: העותק הממופה אינו שווה למקור")
    return FileEntry(rel, file_sig(out / rel), (out / rel).stat().st_size, narrowed.dtype.str, source.dtype.str,
                     list(source.shape), values)


def _write_copy(out: Path, store: EagerTableStore, name: str) -> FileEntry:
    source = store.path(name)
    rel = f"{name}/{source.name}"
    copy_file(source, out / rel)
    return FileEntry(rel, file_sig(out / rel), (out / rel).stat().st_size, None, None, None, None)


def convert(settings: Settings, root: Path, names: list[str], out: Path) -> dict[str, TableEntry]:
    """The derived copy of ``names`` in ``out`` (manifest merged with the tables already there)."""
    store = EagerTableStore(table_refs(settings, root), None)
    shas = store.verify(names)
    planned = plan(store, names, settings.hebzip.narrow_bits)
    new: dict[str, TableEntry] = {}
    for item in planned:
        if item.pairs is None:
            files = {ONLY: _write_copy(out, store, item.name)}
        else:
            files = {k: _write_array(out, item.name, k, s, n) for k, (s, n) in item.pairs.items()}
        new[item.name] = TableEntry(shas[item.name], store.path(item.name).stat().st_size, files)
    write_manifest(out, merged(out, new))
    return new


def _names(settings: Settings, modes: list[str] | None) -> list[str]:
    if not modes:
        return sorted(settings.tables)
    return sorted({t for mode in modes for t in mode_tables(settings, mode)})


def run(config: Path, modes: list[str] | None, out: Path | None) -> list[str]:
    """Convert and return the report lines (table, pinned bytes, derived bytes)."""
    settings = load_config(config)
    root = project_root(config)
    folder = out if out is not None else resolve(root, settings.hebzip.table_store_dir)
    made = convert(settings, root, _names(settings, modes), folder)
    lines = [f"{name}: {e.source_bytes} -> {sum(f.bytes for f in e.files.values())} bytes" for name, e in made.items()]
    return lines + [f"derived copy: {folder}"]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: 0 = every table converted and the manifest written; the error's exit code otherwise."""
    parser = argparse.ArgumentParser(prog="hebzip.ops.convert_tables")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--modes", help="comma-separated mode names (default: every table of the config)")
    parser.add_argument("--out", type=Path, help="write here instead of [hebzip].table_store_dir")
    args = parser.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m] if args.modes else None
    try:
        lines = run(args.config, modes, args.out)
    except HebzipError as error:
        sys.stderr.write(f"{error}\n")
        return error.exit_code
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
