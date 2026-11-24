"""Typed settings from ``config/atar.toml``.

Why strict (the layer table, config row; 3.11 "change without code"): every path, flag and
limit is a row the user can edit; a misspelled field must stop the server now, with a
Hebrew sentence, instead of silently falling back to a default. Per-tool ``options`` are
validated by the drawer that owns them (``compressors.registry``), not here.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atar.core.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "atar.toml"
REPO = ROOT.parents[1]
"""The repository root: the page lives in ``<repo>/tools/compare_page``."""


def _python() -> str:
    return sys.executable


def _repo() -> str:
    return str(REPO)


PLACES: dict[str, Callable[[], str]] = {"{python}": _python, "{repo}": _repo}
"""Words a row may use instead of a machine path: the interpreter running this page, the repository root."""
_SECTIONS = {"server", "limits", "paths", "services", "tools"}
_SERVER = {"host", "port", "open_browser"}
_LIMITS = {"max_text_bytes", "max_upload_bytes", "poll_ms"}
_PATHS = {"material", "logs"}
_TOOL = {"id", "label", "kind", "ours", "timeout_s", "options"}
_SERVICE = {"version", "cwd", "command", "ready_timeout_s"}


@dataclass(frozen=True, slots=True)
class ServiceRow:
    """A tool that stays loaded (``[services.<name>]``): how to start it and how long loading may take.

    Started once in the background when the server starts - only if some row of kind ``service`` uses
    it (``compressors.loaded.ServicePool``) - and restarted on the next use if it died."""

    name: str
    version: str
    cwd: str
    command: tuple[str, ...]
    ready_timeout_s: float


@dataclass(frozen=True, slots=True)
class ToolRow:
    """One table row as configured: who, which drawer, and that drawer's options."""

    id: str
    label: str
    kind: str
    ours: bool
    timeout_s: float
    options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the server needs, already checked."""

    host: str
    port: int
    open_browser: bool
    max_text_bytes: int
    max_upload_bytes: int
    poll_ms: int
    material: Path
    logs: Path
    tools: tuple[ToolRow, ...]
    services: tuple[ServiceRow, ...] = ()


def placed(value: Any) -> Any:
    """``value`` with the ``PLACES`` words filled in - in a string, or in every string of a list.

    Why: the rows that start our own tool must work from any folder and with whatever Python
    runs the page, so the config names neither an absolute path nor an interpreter. ``{in}``, ``{out}``
    and ``{work}`` are left alone (they are filled per call by the drawer)."""
    if isinstance(value, str):
        for word, where in PLACES.items():
            value = value.replace(word, where())
        return value
    if isinstance(value, list):
        return [placed(v) for v in value]
    return value


def keys(table: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(table, dict):
        raise ConfigError(f"בהגדרות: '{where}' צריך להיות טבלה")
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"בהגדרות: שדה לא מוכר ב-'{where}': {', '.join(unknown)}")
    missing = sorted(allowed - set(table) - {"options"})
    if missing:
        raise ConfigError(f"בהגדרות: חסר שדה ב-'{where}': {', '.join(missing)}")
    return table


def typed(value: Any, kind: type, where: str) -> Any:
    if kind is int and isinstance(value, bool):
        raise ConfigError(f"בהגדרות: '{where}' צריך להיות מספר")
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, kind):
        raise ConfigError(f"בהגדרות: '{where}' מסוג לא נכון")
    return value


def tool_row(raw: Any, index: int) -> ToolRow:
    """One `[[tools]]` row, checked (also used by the bench, `atar/bench`)."""
    t = keys(raw, _TOOL, f"tools[{index}]")
    timeout = typed(t["timeout_s"], float, f"tools[{index}].timeout_s")
    if timeout <= 0:
        raise ConfigError(f"בהגדרות: timeout_s של tools[{index}] חייב להיות חיובי")
    return ToolRow(typed(t["id"], str, "tools.id"), typed(t["label"], str, "tools.label"),
                   typed(t["kind"], str, "tools.kind"), typed(t["ours"], bool, "tools.ours"),
                   timeout, {k: placed(v) for k, v in
                             typed(t.get("options", {}), dict, "tools.options").items()})


def _service(name: str, raw: Any) -> ServiceRow:
    where = f"services.{name}"
    s = keys(raw, _SERVICE, where)
    command = placed(typed(s["command"], list, f"{where}.command"))
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ConfigError(f"בהגדרות: '{where}.command' צריך להיות רשימת מחרוזות לא ריקה")
    timeout = typed(s["ready_timeout_s"], float, f"{where}.ready_timeout_s")
    if timeout <= 0:
        raise ConfigError(f"בהגדרות: '{where}.ready_timeout_s' חייב להיות חיובי")
    return ServiceRow(name, typed(s["version"], str, f"{where}.version"),
                      placed(typed(s["cwd"], str, f"{where}.cwd")), tuple(command), timeout)


def service_rows(raw: Any) -> tuple[ServiceRow, ...]:
    """The `[services]` table, checked (also used by the bench)."""
    table = typed(raw, dict, "services")
    return tuple(_service(name, row) for name, row in table.items())


def _path(value: Any, where: str) -> Path:
    p = Path(typed(value, str, where))
    return p if p.is_absolute() else ROOT / p


def parse(data: dict[str, Any]) -> Settings:
    """Validate a parsed TOML document into :class:`Settings`."""
    top = keys(data, _SECTIONS, "קובץ")
    srv, lim = keys(top["server"], _SERVER, "server"), keys(top["limits"], _LIMITS, "limits")
    paths = keys(top["paths"], _PATHS, "paths")
    if srv["host"] != "127.0.0.1":
        raise ConfigError("בהגדרות: host חייב להיות 127.0.0.1 (המחשב הזה בלבד)")
    port = typed(srv["port"], int, "server.port")
    if not 1024 < port < 65536:
        raise ConfigError("בהגדרות: port מחוץ לטווח")
    tools = tuple(tool_row(raw, i) for i, raw in enumerate(typed(top["tools"], list, "tools")))
    ids = [t.id for t in tools]
    if len(set(ids)) != len(ids) or not tools:
        raise ConfigError("בהגדרות: רשימת הכלים ריקה או שיש בה id כפול")
    return Settings(srv["host"], port, typed(srv["open_browser"], bool, "server.open_browser"),
                    typed(lim["max_text_bytes"], int, "limits.max_text_bytes"),
                    typed(lim["max_upload_bytes"], int, "limits.max_upload_bytes"),
                    typed(lim["poll_ms"], int, "limits.poll_ms"),
                    _path(paths["material"], "paths.material"), _path(paths["logs"], "paths.logs"),
                    tools, service_rows(top["services"]))


def load(path: Path = DEFAULT_CONFIG) -> Settings:
    """Read and validate the config file."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"לא הצלחתי לקרוא את קובץ ההגדרות: {type(error).__name__}") from error
    return parse(data)
