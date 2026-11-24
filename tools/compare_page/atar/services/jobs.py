"""One comparison job at a time, on a background thread, with a numbers-only snapshot.

* Never block a request: the upload request only checks and extracts the text, then *starts* the job and
  returns; the page polls :meth:`JobBoard.snapshot` and rows appear as each tool finishes.
* Weak machine (i3, little free memory): tools run one after another, and a second upload
  while busy gets :class:`Busy` -> a polite "עסוק" on the page (one job per asset).
* The machine-and-material rule: the scratch folder is removed in ``finally`` - success, tool failure or
  an unexpected crash alike. The snapshot holds numbers, tool names and the file name to
  show on the page; the file name is never logged.
* Job ids only grow.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atar.compressors.contract import Compressor
from atar.compressors.loaded import ServicePool
from atar.config.load import ToolRow
from atar.core.errors import Busy
from atar.core.logging import get_logger
from atar.domain import report
from atar.domain.extract import extract, word_count
from atar.services import material
from atar.services.runner import run_tool

_log = get_logger("jobs")


@dataclass
class _Job:
    number: int
    file_name: str
    original: int
    words: int
    state: str = "running"  # running | done | error
    running: str = ""
    outcomes: list[report.Outcome] = field(default_factory=list)
    error: str = ""


class JobBoard:
    """Holds the drawers, the material folder and the current (or last) job."""

    def __init__(self, tools: Sequence[tuple[ToolRow, Compressor]], material_dir: Path,
                 max_text_bytes: int, pool: ServicePool | None = None) -> None:
        self.tools = list(tools)
        self.material = material_dir
        self.max_text_bytes = max_text_bytes
        self.pool = pool if pool is not None else ServicePool(())
        self._lock = threading.Lock()
        self._job: _Job | None = None
        self._counter = 0
        self._thread: threading.Thread | None = None

    def start(self, file_name: str, data: bytes) -> int:
        """Check + extract now (fast); run the tools in the background. Raises Busy/InputRejected."""
        with self._lock:
            if self._job is not None and self._job.state == "running":
                raise Busy("עסוק: השוואה אחרת רצה עכשיו. חכה שתסתיים ונסה שוב.")
            raw = extract(file_name, data, self.max_text_bytes)
            self._counter += 1
            job = _Job(self._counter, file_name, len(raw), word_count(raw))
            self._job = job
            folder = material.scratch(self.material, job.number)
            self._thread = threading.Thread(target=self._run, args=(job, folder, raw),
                                            name=f"job-{job.number}", daemon=True)
            self._thread.start()
        _log.info("job started", job=job.number, original=job.original, tools=len(self.tools))
        return job.number

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job thread ends (tests and shutdown only)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, job: _Job, folder: Path, raw: bytes) -> None:
        try:
            text = folder / "text.txt"
            text.write_bytes(raw)
            for row, tool in self.tools:
                with self._lock:
                    job.running = row.label
                outcome = self._one(row, tool, text, folder)
                with self._lock:
                    job.outcomes.append(outcome)
            with self._lock:
                job.state, job.running = "done", ""
        except Exception as error:  # noqa: BLE001 - job-level safety net (last-resort)
            _log.error("job crashed", job=job.number, error=type(error).__name__)
            with self._lock:
                job.state, job.running = "error", ""
                job.error = "תקלה לא צפויה בזמן ההשוואה. אפשר לנסות שוב."
        finally:
            material.remove(folder)
            _log.info("job ended", job=job.number, scratch_left=folder.exists())

    def _one(self, row: ToolRow, tool: Compressor, text: Path, folder: Path) -> report.Outcome:
        try:
            return run_tool(row, tool, text, folder)
        except Exception as error:  # noqa: BLE001 - row-level net: one tool never sinks the job
            _log.error("tool crashed", tool=row.id, error=type(error).__name__)
            return report.Outcome(row.id, row.label, row.ours, None, None, None, None,
                                  "נכשל (תקלה לא צפויה)")

    def services(self) -> list[dict[str, Any]]:
        """The loaded tools' state for the page (loading / ready + load time / failed)."""
        return self.pool.snapshot()

    def versions(self) -> list[dict[str, Any]]:
        """Which tools/versions this server runs (for the note under the table)."""
        return [{"label": row.label, "version": tool.version(), "ours": row.ours}
                for row, tool in self.tools]

    def snapshot(self) -> dict[str, Any]:
        """Everything the page shows - numbers, tool names, the file name; never text."""
        with self._lock:
            job = self._job
            if job is None:
                return {"state": "idle"}
            outcomes = list(job.outcomes)
            done = job.state != "running"
            pending = [row.label for row, _ in self.tools[len(outcomes):]] if not done else []
            return {"state": job.state, "job": job.number, "file_name": job.file_name,
                    "original": job.original, "words": job.words, "running": job.running,
                    "pending": pending, "error": job.error,
                    "rows": report.table(outcomes, job.original),
                    "summary": report.summary(outcomes, done)}
