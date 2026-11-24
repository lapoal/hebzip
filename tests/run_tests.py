"""Run every test FILE in its own pytest process (the one-process-per-test-file rule: shared runs give false failures).

    python tests/run_tests.py            # all files
    python tests/run_tests.py header cli # only files whose path contains one of the words

Exit code 0 only when every file passed. Prints one line per file (this is a tool, so it writes to
stdout directly) and the failing files' pytest output at the end.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILE_TIMEOUT_S = 3600


def run_file(path: Path) -> tuple[bool, str]:
    """pytest on one file in a fresh interpreter -> (passed, tail of output)."""
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(path)], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=FILE_TIMEOUT_S, check=False)
    tail = (done.stdout.strip().splitlines() or ["(no output)"])[-1]
    return done.returncode == 0, done.stdout[-4000:] if done.returncode else tail


def safe_stdout() -> None:
    """Characters stdout's encoding cannot hold are written as ``\\uXXXX``, never raised.

    Why: with the output redirected on Windows, stdout is cp1255; a failing
    file's output (Hebrew errors + the U+FFFD of ``errors="replace"`` above) crashed the runner with
    ``UnicodeEncodeError`` and the other failures were never reported. The encoding is kept (whoever
    reads the log expects it); only the error policy changes. Guard: ``tests/gates/test_runner_report.py``.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")


def main(words: list[str]) -> int:
    safe_stdout()
    files = sorted(p for p in HERE.rglob("test_*.py") if "_work" not in p.parts)
    files = [p for p in files if not words or any(w in str(p) for w in words)]
    failed: list[tuple[Path, str]] = []
    for path in files:
        ok, text = run_file(path)
        sys.stdout.write(f"{'PASS' if ok else 'FAIL'}  {path.relative_to(HERE)}  {text if ok else ''}\n")
        sys.stdout.flush()
        if not ok:
            failed.append((path, text))
    for path, text in failed:
        sys.stdout.write(f"\n===== {path.relative_to(HERE)} =====\n{text}\n")
    sys.stdout.write(f"\n{len(files) - len(failed)}/{len(files)} files passed\n")
    left = work_left()
    sys.stdout.write(f"{'FAIL' if left else 'PASS'}  gate: tests/_work empty after the suite"
                     f"{f' - {len(left)} entries left: {left[:10]}' if left else ''}\n")
    return 1 if failed or left else 0


def work_left() -> list[str]:
    """Names (never contents) of whatever is still in ``tests/_work`` after the suite.

    Gate of the machine-and-material rule: every test removes its scratch folder when it ends
    (``tests/conftest.py``); anything left here is a decoded copy or a package that outlived its test.
    """
    work = HERE / "_work"
    return sorted(p.name for p in work.iterdir()) if work.is_dir() else []


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
