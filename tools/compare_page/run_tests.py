"""The full gate: every test file in its own process, then "no material left", ruff, mypy.

One process per test file (a shared run gives false failures).
Machine and material apart: the suite ends with the gate "tests/_work and material/ are empty".
Usage (from this folder, with a Python that has hebzip's dependencies): python run_tests.py [--fast]
``--fast`` skips the slow real-tools test (test_full_http.py).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLOW = {"test_full_http.py"}


def _say(line: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _pytest(path: Path) -> tuple[int, int]:
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", str(path)], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=False)
    tail = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", tail))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", tail))
    if done.returncode != 0:
        _say(done.stdout[-3000:])
        failed = max(failed, 1)
    _say(f"  {path.name}: {passed} passed, {failed} failed")
    return passed, failed


def _empty(folder: Path) -> bool:
    return not folder.exists() or not any(folder.iterdir())


def main() -> int:
    fast = "--fast" in sys.argv
    files = sorted(p for p in (ROOT / "tests").glob("test_*.py") if not (fast and p.name in SLOW))
    passed = failed = 0
    for path in files:
        p, f = _pytest(path)
        passed, failed = passed + p, failed + f
    gates = {
        "tests/_work empty": _empty(ROOT / "tests" / "_work"),
        "material empty": _empty(ROOT / "material"),
        "ruff": subprocess.run([sys.executable, "-m", "ruff", "check", "atar", "tests", "run_tests.py"],
                               cwd=ROOT, capture_output=True, check=False).returncode == 0,
        "mypy --strict": subprocess.run([sys.executable, "-m", "mypy", "atar", "tests", "run_tests.py"],
                                        cwd=ROOT, capture_output=True, check=False).returncode == 0,
    }
    for name, ok in gates.items():
        _say(f"  gate {name}: {'OK' if ok else 'FAILED'}")
    total = passed + failed
    _say(f"tests: {passed}/{total} passed" + (" (fast: slow test skipped)" if fast else ""))
    return 0 if failed == 0 and all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
