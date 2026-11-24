"""``python -m atar.bench run | report | overlap`` - the one command that re-runs the public measurement.

    python -m atar.bench run [--omilab DIR] [--paq8px paq8px.exe --paq8px-dict DIR]
    python -m atar.bench report
    python -m atar.bench overlap --corpus he=hewiki_random.jsonl --corpus en=enwiki_random.jsonl

``run`` measures every row of ``config/bench.toml`` on every text of ``bench/manifest.json`` (resuming
an interrupted run; ``--fresh`` starts over), then writes ``bench/results/bench.json`` (every pair)
and ``summary.json`` + ``bench.md`` (the tables). Optional inputs: the everyday dataset (its
text is not shipped) and paq8px (GPL, not shipped; ``-8LT`` also needs its english.dic / .exp / .emb,
which are copied next to a scratch copy of the exe because paq8px opens them from its own folder).
Whatever is not given is skipped with a line saying so. Run from ``tools/compare_page``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from atar.bench import markdown, overlap, prior, report, results
from atar.bench.config import DEFAULT_BENCH, BenchSettings, load
from atar.bench.datasets import Rebuilder
from atar.bench.run import Run, run_rows, say
from atar.bench.sets import load_manifest
from atar.compressors.memory import total_ram_mb
from atar.core.clock import stamp
from atar.core.errors import AtarError

PAQ_DICT = ("english.dic", "english.exp", "english.emb")


def _cpu() -> str:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or "unknown CPU"


def machine() -> str:
    """One line naming where the times were measured."""
    return (f"{_cpu()}, {total_ram_mb() / 1024:.1f} GB RAM, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}, measured {stamp()}")


def _paq_copy(args: argparse.Namespace, work: Path) -> str:
    """A scratch copy of paq8px (+ its English files) - the path rows use as ``{paq8px}``."""
    if args.paq8px is None:
        return ""
    folder = work / "paq"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.paq8px, folder / "paq8px.exe")
    for name in PAQ_DICT if args.paq8px_dict else ():
        if (args.paq8px_dict / name).is_file():
            shutil.copyfile(args.paq8px_dict / name, folder / name)
    return str(folder / "paq8px.exe")


def _settings(args: argparse.Namespace) -> BenchSettings:
    first = load(args.config, {})
    paq = _paq_copy(args, first.work)
    return load(args.config, {"{paq8px}": paq or "paq8px-not-given"})


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.languages:
        settings = BenchSettings(settings.manifest, settings.results, settings.work, settings.seen_share,
                                 tuple(args.languages.split(",")), settings.rows, settings.services)
    rows = tuple(r for r in settings.rows if not args.rows or r.tool.id in args.rows.split(","))
    data = {"tools": {}, "rows": []} if args.fresh else results.load(settings.results / "bench.json")
    data["machine"] = machine()
    given = {"paq8px"} if args.paq8px else set()
    run = Run(settings, load_manifest(settings.manifest), data, Rebuilder(args.omilab), given)
    try:
        run_rows(run, rows)
    finally:
        shutil.rmtree(settings.work, ignore_errors=True)
    _write_report(settings, data, args.paq8px_dict)
    return 0


def _write_report(settings: BenchSettings, data: dict[str, Any], paq_dict: Path | None) -> None:
    manifest = load_manifest(settings.manifest)
    found = settings.manifest.parent / "overlap.json"
    seen = json.loads(found.read_text(encoding="utf-8")) if found.is_file() else {}
    summary = report.summary(data, manifest, seen, settings.seen_share)
    sizes = prior.tool_table_bytes(data["tools"])
    summary["prior_table_bytes"] = sizes
    summary["paq8px_dict_bytes"] = prior.dict_files(paq_dict, PAQ_DICT) or _earlier_dict_bytes(settings)
    results.save(settings.results / "summary.json", summary)
    (settings.results / "bench.md").write_text(markdown.render(summary, data, sizes), encoding="utf-8")
    say(f"written: {settings.results / 'bench.md'}")


def _earlier_dict_bytes(settings: BenchSettings) -> dict[str, int]:
    """The paq8px file sizes of the last summary - a ``report`` without --paq8px-dict keeps them
    (they describe the recorded -8LT row, which is still in bench.json)."""
    found = settings.results / "summary.json"
    old = json.loads(found.read_text(encoding="utf-8")) if found.is_file() else {}
    sizes: dict[str, int] = old.get("paq8px_dict_bytes", {})
    return sizes


def cmd_report(args: argparse.Namespace) -> int:
    settings = load(args.config, {})
    _write_report(settings, results.load(settings.results / "bench.json"), args.paq8px_dict)
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    settings = load(args.config, {})
    manifest = load_manifest(settings.manifest)
    rebuild = Rebuilder(args.omilab)
    texts: dict[str, Any] = {}
    for item in args.corpus:
        language, _, path = item.partition("=")
        say(f"overlap: {language} against {Path(path).name} ...")
        texts.update(overlap.check(manifest, language, Path(path), rebuild))
    out = {"what": "training-overlap check (atar/bench/overlap.py)", "window_words": overlap.WINDOW,
           "held_every": overlap.HELD_EVERY, "texts": texts}
    results.save(settings.manifest.parent / "overlap.json", out)
    say(f"written: {settings.manifest.parent / 'overlap.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m atar.bench")
    p.add_argument("command", choices=("run", "report", "overlap"))
    p.add_argument("--config", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--omilab", type=Path, help="folder with token_train.tsv + token_test.tsv")
    p.add_argument("--paq8px", type=Path, help="paq8px.exe (v217 was used)")
    p.add_argument("--paq8px-dict", type=Path, help="folder with english.dic / .exp / .emb (for -8LT)")
    p.add_argument("--languages", help="comma-separated, e.g. he (default: the config's)")
    p.add_argument("--rows", help="comma-separated row ids (default: every row)")
    p.add_argument("--fresh", action="store_true", help="forget earlier results and measure everything again")
    p.add_argument("--corpus", action="append", default=[], help="overlap: LANGUAGE=corpus.jsonl")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"run": cmd_run, "report": cmd_report, "overlap": cmd_overlap}[args.command](args)
    except AtarError as error:
        sys.stderr.write(f"{error}\n")
        return 2
