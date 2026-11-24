"""Cold start (the cold-start rule): a clean interpreter (``-I``: no PYTHONPATH, no user site, no cwd)
can import the entry point and run ``--help``. ``-I`` also drops the cwd from ``sys.path`` and the
package is not installed, so the project root is added inside the ``-c`` string (not in test code).
Also proves import loads no numba and defines no kernel before the engine is chosen."""

from __future__ import annotations

import subprocess
import sys

from tests.support import ROOT

PRELUDE = f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
NOT_AT_IMPORT = ("numba", "hebzip.domain.kernels.rc", "hebzip.domain.kernels.cm", "hebzip.domain.kernels.table")
"""Importing the CLI loads no numba and defines no kernel: the engine row is read
from the config first (``domain/kernels/jit.py``)."""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", "-c", PRELUDE + code], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=120, check=False)


def test_import_entry_point_in_a_clean_interpreter() -> None:
    done = _run("import hebzip.interface.cli")
    assert done.returncode == 0, done.stderr


def test_import_loads_no_numba_and_no_kernel() -> None:
    code = f"import hebzip.interface.cli; bad = [n for n in {NOT_AT_IMPORT!r} if n in sys.modules]; assert not bad, bad"
    done = _run(code)
    assert done.returncode == 0, done.stderr


def test_help_runs_in_a_clean_interpreter() -> None:
    done = _run("from hebzip.interface.cli import main; sys.exit(main(['--help']))")
    assert done.returncode == 0 and "compress" in done.stdout, done.stderr
