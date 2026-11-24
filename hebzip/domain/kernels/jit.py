"""The one switch between "plain Python" and "compiled" for every kernel (design decision).

Why a decorator that is configured BEFORE the kernel modules are imported: numba compiles a kernel
together with the kernels it calls, which it finds by their global names at compile time - so all
of them must be compiled or none. ``services/engines.py`` calls ``configure`` with ``numba.njit``
(engine row ``numba``) or ``None`` (engine row ``python``) and only then imports the kernel
modules. The kernel source is the same in both cases: there is no second copy of any math.
``domain`` stays free of numba: the compiler arrives as an argument.

A process that already defined kernels in one mode cannot switch (``EngineOrderError``) - the
alternative would be a silent mix of compiled and plain kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from hebzip.core.errors import EngineOrderError

Compiler = Callable[[Callable[..., Any]], Callable[..., Any]]

_state: dict[str, Any] = {"compiler": None, "sealed": False, "name": "python"}


def configure(compiler: Compiler | None, name: str) -> None:
    """Choose how kernels are built. Must happen before the first kernel module is imported."""
    if _state["sealed"] and _state["name"] != name:
        raise EngineOrderError(f"kernels were already built as '{_state['name']}', cannot switch to '{name}'")
    _state["compiler"] = compiler
    _state["name"] = name


def kernel[F: Callable[..., Any]](func: F) -> F:
    """Mark ``func`` as a kernel: returned as-is (python) or compiled (numba)."""
    _state["sealed"] = True
    compiler = _state["compiler"]
    return func if compiler is None else cast(F, compiler(func))
