"""A compressor that is an external program: paq8px, or our own CLI (``python -m hebzip``).

Both are called exactly as they are (never modified): the command lines are config rows with
``{in}``, ``{out}`` and ``{work}`` placeholders. The program's own output is captured only to
recognise an out-of-memory failure; it is never logged (it may echo paths).

Memory: paq8px needs ~2.2 GB reserved. If Windows cannot promise
``memory_mb`` right now, the row says "לא רץ (אין מספיק זיכרון)" instead of starting it;
if it still dies with a memory word in its output, the same row status is shown. Never a crash.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from atar.compressors.memory import available_commit_mb
from atar.compressors.stdlib import check_options
from atar.core.errors import ConfigError, NotEnoughMemory, ToolFailed

OPTIONS: dict[str, type] = {"version": str, "cwd": str, "compress": list, "decompress": list,
                            "memory_mb": int, "memory_words": list}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _strings(value: list[Any], where: str) -> tuple[str, ...]:
    if not value or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"בהגדרות: '{where}' צריך להיות רשימת מחרוזות לא ריקה")
    return tuple(value)


class CommandCompressor:
    """Run one configured command per direction, one at a time, with a time budget."""

    def __init__(self, options: dict[str, Any], timeout_s: float,
                 headroom_mb: Callable[[], int] = available_commit_mb) -> None:
        opts = check_options(options, OPTIONS, "command")
        self._version = str(opts["version"])
        self.cwd = str(opts["cwd"])
        self.compress_args = _strings(opts["compress"], "compress")
        self.decompress_args = _strings(opts["decompress"], "decompress")
        self.memory_mb = int(opts["memory_mb"])
        self.memory_words = tuple(w.lower() for w in opts["memory_words"] if isinstance(w, str))
        self.timeout_s = timeout_s
        self.headroom_mb = headroom_mb

    def version(self) -> str:
        return self._version

    def prepare(self) -> None:
        """Started fresh per call: nothing to prepare (the memory check runs per call)."""

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        self._run(self.compress_args, src, dst, work)

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        self._run(self.decompress_args, src, dst, work)

    def _run(self, template: tuple[str, ...], src: Path, dst: Path, work: Path) -> None:
        if self.memory_mb and self.headroom_mb() < self.memory_mb:
            raise NotEnoughMemory()
        fill = {"{in}": str(src), "{out}": str(dst), "{work}": str(work)}
        args = [fill.get(part, part) for part in template]
        cwd = fill.get(self.cwd, self.cwd)
        try:
            done = subprocess.run(args, cwd=cwd, capture_output=True, timeout=self.timeout_s,
                                  stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW, check=False)
        except subprocess.TimeoutExpired as error:
            raise ToolFailed("הכלי חרג מהזמן שהוקצב לו", "נעצר (חרג מהזמן)") from error
        except OSError as error:
            raise ToolFailed("התוכנה לא נמצאה או לא נפתחה", "נכשל (התוכנה לא נמצאה)") from error
        if done.returncode != 0:
            said = (done.stdout + done.stderr).decode("utf-8", "replace").lower()
            if any(word in said for word in self.memory_words):
                raise NotEnoughMemory()
            raise ToolFailed(f"הכלי נכשל (קוד יציאה {done.returncode})")
        if not dst.is_file():
            raise ToolFailed("הכלי לא כתב קובץ פלט")
