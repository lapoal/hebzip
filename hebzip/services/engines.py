"""The engine drawers: build the kernels as plain Python or compiled by numba (design decision).

Why here: the engine is chosen from the config, so it must happen after the config is read and
BEFORE any kernel module is imported (``domain/kernels/jit.py``). This file is the only importer
of the kernel modules and of numba. numba missing -> plain Python + a log line (the accelerator-is-never-a-
dependency rule:
the accelerator is never a dependency - the same functions run either way, the bytes are equal).
numba's compile cache goes to ``<project>/.cache/numba`` (set before numba is imported).
Why two functions and no ``if engine == "numba"``: each engine word is a key of the
selector's table (``REAL["engine"]`` in ``services/select.py``) pointing at one of the compiler
functions below; ``kernels`` only builds with what it is handed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hebzip.core.logging import get_logger
from hebzip.domain.contracts import KernelSet
from hebzip.domain.kernels import jit

LOG = get_logger("engine")
Compiled = tuple[jit.Compiler | None, str]
"""(the compiler or ``None`` for plain Python, the engine name actually used)."""
_built: dict[str, KernelSet] = {}


def python_compiler(root: Path) -> Compiled:
    """The engine ``python``: the kernels run as written."""
    return None, "python"


def numba_compiler(root: Path) -> Compiled:
    """The engine ``numba``: ``numba.njit(cache=True)``; numba missing -> the python engine + a log line."""
    os.environ.setdefault("NUMBA_CACHE_DIR", str(root / ".cache" / "numba"))
    try:
        numba: Any = importlib.import_module("numba")
    except ImportError:
        LOG.warning("engine_fallback", wanted="numba", used="python")
        return python_compiler(root)
    compiler: jit.Compiler = numba.njit(cache=True)
    return compiler, "numba"


def kernels(engine: str, compile_with: Callable[[Path], Compiled], root: Path) -> KernelSet:
    """The kernels for the engine word ``engine`` (built once per process with ``compile_with``)."""
    if engine in _built:
        return _built[engine]
    compiler, name = compile_with(root)
    jit.configure(compiler, name)
    rc: Any = importlib.import_module("hebzip.domain.kernels.rc")
    cm: Any = importlib.import_module("hebzip.domain.kernels.cm")
    table: Any = importlib.import_module("hebzip.domain.kernels.table")
    built = KernelSet(name, rc.enc_bit, rc.dec_bit, rc.enc_symbol, rc.dec_target, rc.dec_symbol, rc.dec_start,
                      cm.cm_byte, cm.flat_byte, cm.rehash, table.table_value, table.table_find,
                      cm.seed, cm.roll, cm.finish, cm.word_ctx)
    _built[engine] = built
    return built
