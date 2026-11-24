"""The website's Python side: the worker calls these four functions (copied to site/py/entry.py by the build).

Why so thin: the page must run exactly the code the CLI runs - the same ``Assembly``, ``compress_bytes`` and
``decompress_bytes`` - so a package made in the browser is the package the CLI makes. Only the glue lives here:
bytes in from JavaScript, bytes out, and errors as their class name (the page shows its own English text;
the tool's messages are never shown, and they never carry the user's text anyway).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hebzip.config.load import load_config, project_root
from hebzip.domain.header import parse_header
from hebzip.services.compress import compress_bytes
from hebzip.services.decompress import decompress_bytes
from hebzip.services.select import Assembly

_ASM: list[Assembly] = []
"""The one assembly of this page (built by ``boot``)."""


def _asm() -> Assembly:
    return _ASM[0]


def _bytes(data: Any) -> bytes:
    """A JavaScript Uint8Array (a Pyodide proxy) or Python bytes -> bytes."""
    return bytes(data.to_bytes()) if hasattr(data, "to_bytes") else bytes(data)


def boot(config: str) -> None:
    """Read and check the browser config (made by ``hebzip.ops.config_subset``)."""
    _ASM[:] = [Assembly(load_config(Path(config)), project_root())]


def prepare(mode: str) -> None:
    """Check the mode's tables against their sha256 and build its drawers (done once, before the first text)."""
    _asm().parts(mode)


def compress(data: Any, mode: str) -> bytes:
    """Text bytes (UTF-8) -> the package, no fallback (the page decompresses it itself to verify)."""
    return compress_bytes(_asm(), _bytes(data), mode, use_fallback=False).data


def decompress(data: Any) -> bytes:
    """A package -> the original bytes."""
    return decompress_bytes(_asm(), _bytes(data))


def mode_of(data: Any) -> str:
    """The mode a package was made with; ``id:<n>`` when the website does not run that mode (the page names it
    from ``build.json``'s list of every mode of the full tool)."""
    header = parse_header(_bytes(data))
    return _asm().mode_by_id(header.mode_id) or f"id:{header.mode_id}"
