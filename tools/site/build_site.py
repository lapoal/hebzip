"""Assemble the generated parts of the website (``site/``) from the repository - so the page never drifts.

    python tools/site/build_site.py --pyodide <folder of the Pyodide 314.0.8 release>
    python tools/site/build_site.py            # Pyodide already in site/pyodide: only checked, not copied
    python tools/site/build_site.py --check    # build into a scratch folder and compare with site/ (exit 1 = drift)

Why generated: the browser runs the CURRENT code of ``hebzip/``, the config rows of mode ``fast``
and its pinned tables. A hand-made copy (the feasibility check had one) goes stale the day a row or a table
changes. Everything here comes from one place each, and ``site/build.json`` records every file the page
fetches with its size and sha256 - the page shows the download size from it and checks each file it fetched.
Written by hand and NOT touched here: ``site/index.html``, ``site/assets/*``, ``site/settings.json``.

    site/py/hebzip.zip       every .py of the package (fixed timestamps: the same code = the same zip)
    site/py/browser.toml     ``hebzip.ops.config_subset`` of the real config (proven to sign like it)
    site/tables/             the pinned tables of the modes (sha256 checked) + their LICENSE / CREDITS
    site/pyodide/            the Pyodide files + numpy wheel, unmodified, + their licence notices
    site/data/bench.json     the recorded benchmark (numbers only) from the bench results
This is a tool: it reports to stdout directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:  # a tool run as a script: make the repository's own package importable
    sys.path.insert(0, str(REPO))

from bench_data import bench_for_site  # noqa: E402  (tools/site/bench_data.py)
from table_files import entry, unpack, write_table  # noqa: E402  (tools/site/table_files.py)

from hebzip.config.load import DEFAULT_CONFIG, load_config  # noqa: E402
from hebzip.config.tables import mode_tables  # noqa: E402
from hebzip.ops.config_subset import make  # noqa: E402

ZIP_TIME = (2026, 1, 1, 0, 0, 0)
"""Every zip member gets this timestamp: the zip depends on the code only."""
GENERATED = ("py", "tables", "data", "samples")
"""The folders this script owns completely (rebuilt from scratch each time); ``pyodide`` is copied file by file."""


class BuildError(Exception):
    """A build input is missing or does not match its pin - nothing half-built is left as the site."""


@dataclass(frozen=True)
class Plan:
    """build.toml, read."""

    out: Path
    modes: list[str]
    package: str
    table_docs: list[str]
    rows: dict[str, Any]
    samples: dict[str, str]
    pyodide: dict[str, Any]
    bench: Path


def read_plan(path: Path = HERE / "build.toml", out: Path | None = None) -> Plan:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    site = data["site"]
    return Plan(out or REPO / site["out"], list(site["modes"]), site["package"], list(site["table_docs"]),
                dict(site["rows"]), dict(site["samples"]), dict(data["pyodide"]), REPO / data["bench"]["results"])


def toml_literal(value: Any) -> str:
    """A row value as TOML text (strings, booleans, integers, lists of those)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(v) for v in value) + "]"
    raise BuildError(f"build.toml: a row value of type {type(value).__name__} is not supported")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_code_zip(plan: Plan, dest: Path) -> None:
    """Every ``.py`` of the package, sorted, fixed timestamps, deflated."""
    files = sorted(p for p in (REPO / plan.package).rglob("*.py") if "__pycache__" not in p.parts)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(REPO).as_posix(), ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())


def write_config(plan: Plan, dest: Path) -> None:
    rows = {name: toml_literal(value) for name, value in plan.rows.items()}
    dest.write_text(make(DEFAULT_CONFIG, plan.modes, rows), encoding="utf-8", newline="\n")


def copy_tables(plan: Plan, dest: Path) -> list[str]:
    """The pinned tables of the modes, each checked against its sha256 row before it is copied."""
    settings = load_config(DEFAULT_CONFIG)
    names = sorted({t for m in plan.modes for t in mode_tables(settings, m)})
    for name in names:
        row = settings.tables[name]
        source = REPO / row.path
        if not source.is_file():
            raise BuildError(f"table {name}: {row.path} is missing (see tables/README.md)")
        if sha256(source) != row.sha256:
            raise BuildError(f"table {name}: {row.path} is not the pinned file (sha256 differs)")
        write_table(source.read_bytes(), dest.parent / row.path)
    for doc in plan.table_docs:
        shutil.copyfile(REPO / doc, dest / Path(doc).name)
    return names


def pyodide_files(plan: Plan, source: Path) -> list[str]:
    """The Pyodide file names the site needs: the core files + the wheels the lock names for ``packages``."""
    lock = json.loads((source / "pyodide-lock.json").read_text(encoding="utf-8"))
    wheels = [lock["packages"][p]["file_name"] for p in plan.pyodide["packages"]]
    return [*plan.pyodide["files"], *wheels]


def copy_pyodide(plan: Plan, source: Path | None) -> None:
    """Copy the Pyodide files unmodified (or check those already in site/pyodide) + the licence notices."""
    dest = plan.out / "pyodide"
    origin = source or dest
    if not (origin / "pyodide-lock.json").is_file():
        raise BuildError("Pyodide is not in site/pyodide - give --pyodide <folder of the 314.0.8 release>")
    dest.mkdir(parents=True, exist_ok=True)
    for name in pyodide_files(plan, origin):
        if not (origin / name).is_file():
            raise BuildError(f"Pyodide file {name} is missing in the given folder")
        if origin != dest:
            shutil.copyfile(origin / name, dest / name)
    for notice in plan.pyodide["notices"]:
        shutil.copyfile(REPO / notice, dest / Path(notice).name)


def write_bench(plan: Plan, dest: Path) -> None:
    summary = json.loads((plan.bench / "summary.json").read_text(encoding="utf-8"))
    data = json.loads((plan.bench / "bench.json").read_text(encoding="utf-8"))
    out = bench_for_site(summary, data)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def fetched_files(plan: Plan, tables: list[str]) -> list[str]:
    """Every file the page fetches, relative to the site (the worker reads the same list from build.json)."""
    settings = load_config(DEFAULT_CONFIG)
    origin = plan.out / "pyodide"
    return ["py/hebzip.zip", "py/browser.toml", "py/entry.py", *sorted(settings.tables[t].path for t in tables),
            "data/bench.json", *(f"pyodide/{n}" for n in pyodide_files(plan, origin))]


def manifest(plan: Plan, tables: list[str]) -> dict[str, Any]:
    """Every fetched file with its size and sha256 (the page's progress bar and its integrity check)."""
    files = {name: entry(plan.out, name) for name in fetched_files(plan, tables)}
    settings = load_config(DEFAULT_CONFIG)
    packs = {m: sorted(settings.tables[t].path for t in mode_tables(settings, m)) for m in plan.modes}
    every = {str(row.id): name for name, row in settings.modes.items()}
    return {"what": "generated by tools/site/build_site.py - do not edit", "modes": plan.modes, "packs": packs,
            "all_modes": every,
            "tables": tables,
            "pyodide": plan.pyodide["version"], "files": files,
            "total_bytes": sum(f["bytes"] for f in files.values()),
            "total_raw_bytes": sum(f.get("raw_bytes", f["bytes"]) for f in files.values())}


def build(plan: Plan, pyodide: Path | None) -> dict[str, Any]:
    """Rebuild every generated part of ``plan.out``; returns the manifest written to build.json."""
    plan.out.mkdir(parents=True, exist_ok=True)
    for folder in GENERATED:
        shutil.rmtree(plan.out / folder, ignore_errors=True)
        (plan.out / folder).mkdir()
    write_code_zip(plan, plan.out / "py" / "hebzip.zip")
    write_config(plan, plan.out / "py" / "browser.toml")
    shutil.copyfile(HERE / "entry.py", plan.out / "py" / "entry.py")
    for language, source in plan.samples.items():
        shutil.copyfile(REPO / source, plan.out / "samples" / f"{language}.txt")
    tables = copy_tables(plan, plan.out / "tables")
    copy_pyodide(plan, pyodide)
    write_bench(plan, plan.out / "data" / "bench.json")
    out = manifest(plan, tables)
    (plan.out / "build.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def check(plan: Plan, scratch: Path) -> list[str]:
    """Build into ``scratch`` (Pyodide taken from the real site) and list every generated file that differs."""
    trial = Plan(scratch, plan.modes, plan.package, plan.table_docs, plan.rows, plan.samples, plan.pyodide,
                 plan.bench)
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        build(trial, plan.out / "pyodide")
        names = sorted({*_generated(plan.out), *_generated(scratch)})
        return [n for n in names if not (plan.out / n).is_file() or not (scratch / n).is_file()
                or sha256(plan.out / n) != sha256(scratch / n)]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _generated(root: Path) -> list[str]:
    files = [p for folder in (*GENERATED, "pyodide") for p in (root / folder).rglob("*") if p.is_file()]
    return [p.relative_to(root).as_posix() for p in files] + ["build.json"]


def unpack_tables(plan: Plan) -> list[str]:
    """The tables of the website's modes -> the project's ``tables/`` (a fresh clone: no download needed)."""
    settings = load_config(DEFAULT_CONFIG)
    names = {t for m in plan.modes for t in mode_tables(settings, m)}
    rows = {n: (settings.tables[n].path, settings.tables[n].sha256) for n in names}
    return unpack(plan.out, rows, REPO)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools/site/build_site.py")
    parser.add_argument("--pyodide", type=Path, help="folder of the Pyodide release (default: site/pyodide)")
    parser.add_argument("--check", action="store_true", help="compare a fresh build with site/; change nothing")
    parser.add_argument("--unpack-tables", action="store_true",
                        help="write the website's tables (modes fast, en_fast) into tables/ for the native tool")
    args = parser.parse_args(argv)
    plan = read_plan()
    try:
        if args.unpack_tables:
            written = unpack_tables(plan)
            sys.stdout.write(f"tables written into tables/: {len(written)} files, each checked against its sha256\n")
            return 0
        if args.check:
            drift = check(plan, REPO / "tests" / "_work" / "site_check")
            sys.stdout.write("site/ is up to date\n" if not drift else "differs: " + ", ".join(drift) + "\n")
            return 1 if drift else 0
        out = build(plan, args.pyodide)
    except (BuildError, OSError, KeyError, ValueError) as error:
        sys.stderr.write(f"build failed: {error}\n")
        return 2
    mb = out["total_bytes"] / 2**20
    sys.stdout.write(f"site built: {len(out['files'])} files, {mb:.1f} MB fetched by the page\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
