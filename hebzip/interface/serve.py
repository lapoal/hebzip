"""The loaded service: ``python -m hebzip serve [--modes max,max_plus] [--config PATH]``.

Why: a CLI call spends ~3 s on a short text, of which the coding is ~0.1 s - the rest is the interpreter,
the imports, building the mode and loading numba's cache, paid again on every call. The comparison page
keeps this process instead: it builds the modes once, then answers requests one after another.

Protocol - JSON lines over stdin / stdout, ASCII only (``ensure_ascii``), one object per line:
* when ready (once): ``{"ready": true, "modes": [...], "load_s": 2.4, "pid": 123}``; or, if it cannot
  start, ``{"ready": false, "exit": 2, "error": "..."}`` and the process ends with that exit code.
* requests: ``{"id": 7, "op": "compress", "input": PATH, "output": PATH, "mode": "max", "fallback": true,
  "force": true}`` · ``{"id": 8, "op": "decompress", "input": PATH, "output": PATH, "force": true}`` ·
  ``{"id": 9, "op": "ping"}``. Only file PATHS cross the pipe - never text or package bytes.
* replies: ``{"id": 7, "ok": true, "exit": 0, "mode": "max", "before": 1551, "after": 248, "seconds": 0.2}``
  or ``{"id": 7, "ok": false, "exit": 7, "error": "<Hebrew>", "error_type": "InputError"}`` - the exit
  codes are the CLI's. A failed request never ends the service.
* stdin closed (the parent ended, or closed it) -> the service ends with exit 0; a reply that cannot be
  written (the parent is gone) ends it too. No other way out, no timer.

Same bytes as the CLI: both call ``run_files.compress_file`` / ``decompress_file`` (one flow); the only
difference is that the ``Assembly`` (modes, tables, kernels) is kept. Every drawer restarts its state at
the start of each message (``start()``), so a package depends only on its own text, and a decode only on
its package bytes (the real-path rule and the one-process-per-test-file rule) - proven by
``tests/interface/test_serve.py``.
Warm-up: each requested mode compresses and decodes one fixed sentence written here (never user text), so
the first real request does not pay for numba's cache or the dictionary's index.
Memory between requests (stated, the machine-and-material rule): no text, no package, no file name is kept. Two speed
caches stay filled while the process lives (``spell.chains`` letter-context memo, ``spell.speller`` tree
nodes): keys are numbers derived from letter contexts / dictionary ranges that were coded; nothing of them
is ever written to disk or to the log, and they die with the process.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from hebzip.core.clock import monotonic
from hebzip.core.errors import BUG_KINDS, HebzipError, InputError
from hebzip.core.logging import get_logger
from hebzip.interface.run_files import UNEXPECTED, Done, assembly, compress_file, decompress_file
from hebzip.services.compress import compress_bytes
from hebzip.services.decompress import decompress_bytes
from hebzip.services.select import Assembly

LOG = get_logger("serve")
WARM_TEXT = "שלום. זהו משפט חימום קצר, שנכתב בתוך הקוד.\n"
NO_MEMORY = "אין מספיק זיכרון לבקשה הזאת"
LOAD_ERRORS: tuple[type[BaseException], ...] = (*BUG_KINDS, MemoryError)
"""What the loading step catches besides the typed errors (then: not-ready line + exit 1)."""
Reply = dict[str, object]


_REQUIRED = object()


def _field(request: dict[str, Any], name: str, kinds: tuple[type, ...], default: object = _REQUIRED) -> Any:
    """One request field of an allowed type (``default`` when absent; no default = required).
    Missing or of another type -> ``InputError`` naming the field, never its value."""
    value = request.get(name, default)
    if not isinstance(value, kinds):
        raise InputError(f"בקשה לא תקינה: השדה '{name}' חסר או מסוג לא נכון")
    return value


def _compress(asm: Assembly, request: dict[str, Any]) -> Done:
    return compress_file(asm, Path(_field(request, "input", (str,))), Path(_field(request, "output", (str,))),
                         _field(request, "mode", (str, type(None)), None), _field(request, "fallback", (bool,), True),
                         _field(request, "force", (bool,), False))


def _decompress(asm: Assembly, request: dict[str, Any]) -> Done:
    return decompress_file(asm, Path(_field(request, "input", (str,))), Path(_field(request, "output", (str,))),
                           _field(request, "force", (bool,), False))


def _ping(asm: Assembly, request: dict[str, Any]) -> Done:
    del asm, request
    return Done(0, 0, "", 0.0)


OPS: dict[str, Callable[[Assembly, dict[str, Any]], Done]] = {
    "compress": _compress, "decompress": _decompress, "ping": _ping}
"""Request word -> what it runs (a table, not a branch)."""


def _parse(line: bytes) -> dict[str, Any]:
    try:
        request = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputError("בקשה לא תקינה: זו לא שורת JSON") from error
    if not isinstance(request, dict) or request.get("op") not in OPS:
        raise InputError("בקשה לא תקינה: חסרה פעולה מוכרת (op)")
    return request


def _failure(rid: object, exit_code: int, message: str, kind: str) -> Reply:
    return {"id": rid, "ok": False, "exit": exit_code, "error": message, "error_type": kind}


def answer(asm: Assembly, line: bytes) -> Reply:
    """One request line -> one reply; any failure is a reply, never the end of the service."""
    rid: object = None
    try:
        request = _parse(line)
        rid = request.get("id")
        done = OPS[request["op"]](asm, request)
    except HebzipError as error:
        return _failure(rid, error.exit_code, str(error), type(error).__name__)
    except MemoryError:
        LOG.error("out of memory")
        return _failure(rid, 1, NO_MEMORY, "MemoryError")
    except BUG_KINDS as error:
        LOG.error("unexpected failure")
        return _failure(rid, 1, UNEXPECTED, type(error).__name__)
    return {"id": rid, "ok": True, "exit": 0, **asdict(done)}


def _send(out: TextIO, message: Reply) -> bool:
    """One line to the parent; ``False`` when it cannot be written (the parent is gone)."""
    try:
        out.write(json.dumps(message, ensure_ascii=True) + "\n")
        out.flush()
    except (OSError, ValueError):
        return False
    return True


def _load(config: Path, modes: list[str] | None) -> tuple[Assembly, list[str]]:
    """Build every requested mode and run the fixed warm-up sentence through it (compress + decode)."""
    asm = assembly(config)
    names = modes or [asm.settings.hebzip.default_mode]
    for mode in names:
        package = compress_bytes(asm, WARM_TEXT.encode("utf-8"), mode).data
        decompress_bytes(asm, package)
    return asm, names


def _start(config: Path, modes: list[str] | None, out: TextIO) -> Assembly | int:
    """The loaded assembly after the ready line, or the exit code after the not-ready line."""
    started = monotonic()
    try:
        asm, names = _load(config, modes)
    except HebzipError as error:
        _send(out, {"ready": False, "exit": error.exit_code, "error": str(error)})
        return error.exit_code
    except LOAD_ERRORS:
        LOG.error("unexpected failure while loading")
        _send(out, {"ready": False, "exit": 1, "error": UNEXPECTED})
        return 1
    load_s = round(monotonic() - started, 2)
    LOG.info("service ready", modes=len(names), load_s=load_s)
    if not _send(out, {"ready": True, "modes": names, "load_s": load_s, "pid": os.getpid()}):
        return 0
    return asm


def run(config: Path, modes: list[str] | None, source: BinaryIO, out: TextIO) -> int:
    """Load, say ready, then answer line after line until stdin closes."""
    loaded = _start(config, modes, out)
    if isinstance(loaded, int):
        return loaded
    count = 0
    for line in iter(source.readline, b""):
        if not line.strip():
            continue
        count += 1
        if not _send(out, answer(loaded, line)):
            break
    LOG.info("service stopped", requests=count)
    return 0
