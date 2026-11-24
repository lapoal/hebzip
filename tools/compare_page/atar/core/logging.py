"""One structured logger - never handed text, and never the uploaded file name.

Namespace ``atar``.
The no-user-content-in-metrics rule and the machine-and-material rule: logs hold numbers and tool names
only. The secret scrubber was
dropped - this tool holds no secrets; the discipline is "never pass text in", and a
test greps the log after a real job for a canary word.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from atar.core.clock import stamp

# Until the server configures its file, records go nowhere (never to stderr).
logging.getLogger("atar").addHandler(logging.NullHandler())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        payload = {"ts": stamp(), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage(), **fields}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class Logger:
    """Thin wrapper: ``log.info("msg", tool="xz", bytes=12)`` - fields must be scalars."""

    def __init__(self, name: str) -> None:
        self._inner = logging.getLogger(name)

    def _emit(self, level: int, msg: str, fields: dict[str, object]) -> None:
        self._inner.log(level, msg, extra={"fields": fields})

    def info(self, msg: str, **fields: object) -> None:
        self._emit(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: object) -> None:
        self._emit(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: object) -> None:
        self._emit(logging.ERROR, msg, fields)


def get_logger(name: str) -> Logger:
    """Return the structured logger for one module."""
    return Logger(f"atar.{name}")


def configure_file_logging(path: Path) -> None:
    """Send every ``atar.*`` record to one JSON-lines file. Idempotent."""
    root = logging.getLogger("atar")
    root.setLevel(logging.INFO)
    root.propagate = False
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)


def stop_file_logging() -> None:
    """Close every file handler (Windows cannot delete a log file that is still open)."""
    root = logging.getLogger("atar")
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
