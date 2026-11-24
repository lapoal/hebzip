"""One structured logger: JSON lines on stderr, never the user's text.

Root name ``hebzip``;
no secret scrubbing (hebzip has no secrets and never opens the network); the only sink is
stderr, configured once by the CLI (``configure_stderr``). Until then records go nowhere, so a
library user and the tests see no stray output. Fields must be numbers, names or short hashes
(the no-user-content-in-metrics rule: metrics never hold content).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Final

ROOT: Final = "hebzip"
LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR")
MAX_FRAMES: Final = 12

logging.getLogger(ROOT).addHandler(logging.NullHandler())


class JsonFormatter(logging.Formatter):
    """One JSON object per line: level, logger, msg + the call's fields."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        payload: dict[str, object] = {
            "level": record.levelname, "logger": record.name, "msg": record.getMessage(), **fields}
        if record.exc_info and record.exc_info[0] is not None:
            payload["error_type"] = record.exc_info[0].__name__
            #: the traceback's places (never its message - it may quote the user's text)
            frames = traceback.extract_tb(record.exc_info[2])[-MAX_FRAMES:]
            payload["where"] = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """The one stderr sink; a class of its own so ``configure_stderr`` can find it again."""


class Logger:
    """Thin wrapper: ``log.info("msg", mode="fast", bytes=12)`` - fields are scalars only."""

    def __init__(self, name: str) -> None:
        self._inner = logging.getLogger(name)

    def _emit(self, level: int, msg: str, fields: dict[str, object], loud: bool = False) -> None:
        self._inner.log(level, msg, extra={"fields": fields}, exc_info=loud)

    def debug(self, msg: str, **fields: object) -> None:
        """Detail for debugging."""
        self._emit(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: object) -> None:
        """A normal event."""
        self._emit(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: object) -> None:
        """Something went wrong but the run goes on."""
        self._emit(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: object) -> None:
        """A failure the user must hear about; carries the exception type when one is active."""
        self._emit(logging.ERROR, msg, fields, loud=sys.exc_info()[0] is not None)


def get_logger(name: str) -> Logger:
    """The structured logger of one module (``hebzip.<name>``)."""
    return Logger(f"{ROOT}.{name}")


def configure_stderr(level: str) -> None:
    """Send every ``hebzip.*`` record to stderr as JSON lines. Idempotent (same handler kept)."""
    root = logging.getLogger(ROOT)
    root.setLevel(level if level in LEVELS else "INFO")
    root.propagate = False
    if any(isinstance(h, StderrHandler) for h in root.handlers):
        return
    handler = StderrHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
