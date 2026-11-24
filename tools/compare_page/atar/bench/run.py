"""The run: every row x every text of its languages, saved after each text, resumable.

Order and memory (a weak laptop): rows run one after another, never in parallel; a loaded
service is started by its first row and stopped right after its last row, so it never sits in memory
next to paq8px (which reserves ~2.2 GB). Every pair gets a fresh folder in the work folder, removed
after the pair; the work folder itself (dataset texts, paq8px's copy) is removed at the end, also
after a failure (the machine-and-material rule).
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atar.bench import results
from atar.bench.config import BenchRow, BenchSettings
from atar.bench.datasets import Rebuilder
from atar.bench.measure import measure
from atar.bench.sets import Manifest, TextRef, TextSet, text_bytes
from atar.compressors.contract import Compressor
from atar.compressors.loaded import ServicePool
from atar.compressors.registry import build
from atar.config.load import REPO
from atar.core.errors import AtarError

Say = Callable[[str], None]


def say(line: str) -> None:
    """Progress to stdout (this is a tool: it reports directly)."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


@dataclass
class Run:
    """One bench run's state."""

    settings: BenchSettings
    manifest: Manifest
    data: dict[str, Any]
    rebuild: Rebuilder
    given: set[str]
    say: Say = say
    texts: dict[str, Path] = field(default_factory=dict)
    measured_on: str = ""
    """The machine line of this run, kept with every row that measures a text in it (``cli`` module text)."""

    def wanted(self, row: BenchRow) -> list[TextSet]:
        """The sets this row runs on (language of the row and of the run; dataset given)."""
        return [s for s in self.manifest.sets
                if s.language in row.languages and s.language in self.settings.languages
                and (not s.dataset or self.rebuild.has(s.dataset))]

    def text_path(self, ref: TextRef) -> Path:
        """The text's file: in place when shipped, else rebuilt into the work folder (checked)."""
        if ref.name not in self.texts:
            if ref.file:
                text_bytes(ref, None)
                self.texts[ref.name] = REPO / ref.file
            else:
                path = self.settings.work / "texts" / f"{ref.name}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text_bytes(ref, self.rebuild.text))
                self.texts[ref.name] = path
        return self.texts[ref.name]


def _skipped_sets(run: Run) -> None:
    for s in run.manifest.sets:
        if s.dataset and not run.rebuild.has(s.dataset):
            run.say(f"skip set {s.id}: dataset '{s.dataset}' not given (--{s.dataset} PATH)")


def _one_row(run: Run, row: BenchRow, tool: Compressor) -> None:
    done = results.done_pairs(run.data)
    for text_set in run.wanted(row):
        for ref in text_set.texts:
            if (row.tool.id, ref.name) in done:
                continue
            folder = run.settings.work / "pair"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)
            try:
                got = measure(row, tool, text_set.id, ref.name, run.text_path(ref), folder)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            results.put(run.data, got.as_json())
            if run.measured_on:
                run.data["tools"][row.tool.id]["measured_on"] = run.measured_on
            results.save(run.settings.results / "bench.json", run.data)
            run.say(f"{row.tool.id:14} {text_set.id:20} {ref.name:34} {got.raw:6} -> {got.bytes:6}  "
                    f"{got.status}  c={got.compress_s}s d={got.decompress_s}s")


def _service_of(row: BenchRow) -> str:
    return str(row.tool.options.get("service", "")) if row.tool.kind == "service" else ""


def run_rows(run: Run, rows: tuple[BenchRow, ...]) -> None:
    """Every row in order; a service stops after the last row that uses it."""
    pool = ServicePool(run.settings.services)
    _skipped_sets(run)
    try:
        for i, row in enumerate(rows):
            if row.needs and row.needs not in run.given:
                run.say(f"skip row {row.tool.id}: needs --{row.needs}")
                continue
            tool = build(row.tool, pool)
            earlier = run.data["tools"].get(row.tool.id, {}).get("measured_on")
            run.data["tools"][row.tool.id] = _tool_info(row, tool) | (
                {"measured_on": earlier} if earlier else {})
            try:
                _one_row(run, row, tool)
            except AtarError as error:
                run.say(f"row {row.tool.id} stopped: {error}")
            name = _service_of(row)
            if name:
                run.data["tools"][row.tool.id]["service_load_s"] = pool.used[name].load_s
            if name and all(_service_of(later) != name for later in rows[i + 1:]):
                pool.used[name].stop()
    finally:
        pool.stop_all()


def _place(value: object, here: str) -> str:
    """A path as the published results may hold it: '' when it is this repository / this Python.

    Why: the results file is public; it used to record the absolute
    folder and interpreter of the machine that ran the bench. '' means "the copy the report runs in",
    which is what ``prior.tool_table_bytes`` falls back to - the same answer, no machine path."""
    text = str(value)
    return "" if text in ("", here) else text


def _tool_info(row: BenchRow, tool: Compressor) -> dict[str, Any]:
    """What the report says about the tool. For our rows also WHICH hebzip (its folder and Python, from the
    row's CLI options) - the prior-knowledge size is asked from that copy ('' = this one, ``_place``)."""
    command = row.tool.options.get("compress") or [""]
    return {"label": row.tool.label, "kind": row.tool.kind, "ours": row.tool.ours, "version": tool.version(),
            "header_bytes": row.header_bytes, "prior": row.prior, "languages": list(row.languages),
            "hebzip_mode": row.hebzip_mode, "repo": _place(row.tool.options.get("cwd", ""), str(REPO)),
            "python": _place(command[0], sys.executable)}
