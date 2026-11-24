"""Typed settings of the bench, from ``config/bench.toml``.

Why the page's own row format: a bench row IS a page row (``[[tools]]``: id, label, kind, ours,
timeout_s, options) plus what only a published table needs - the fixed per-package header to
subtract, the languages it runs on, the prior knowledge it ships with, and the external program it
needs (``paq8px``: GPL, not shipped, so its rows run only when its folder is given). The page row is
checked by the page's own parser (``config.load.tool_row``), so the two cannot drift; an unknown
field stops the bench at start-up, naming it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atar.config.load import ROOT, ServiceRow, ToolRow, keys, service_rows, tool_row, typed
from atar.core.errors import ConfigError

DEFAULT_BENCH = ROOT / "config" / "bench.toml"
_SECTIONS = {"bench", "services", "rows"}
_BENCH = {"manifest", "results", "work", "seen_share", "languages"}
_ROW_EXTRA = {"header_bytes", "languages", "prior", "needs", "hebzip_mode", "mode_id"}
NEEDS = ("paq8px",)
"""External programs a row may need; each is given on the command line (``--paq8px DIR``)."""


@dataclass(frozen=True, slots=True)
class BenchRow:
    """One tool row of the published table."""

    tool: ToolRow
    header_bytes: int
    languages: tuple[str, ...]
    prior: str
    needs: str
    hebzip_mode: str
    mode_id: int


@dataclass(frozen=True, slots=True)
class BenchSettings:
    """Everything the bench needs, already checked."""

    manifest: Path
    results: Path
    work: Path
    seen_share: float
    languages: tuple[str, ...]
    rows: tuple[BenchRow, ...]
    services: tuple[ServiceRow, ...]


def _path(value: Any, where: str) -> Path:
    path = Path(typed(value, str, where))
    return path if path.is_absolute() else ROOT / path


def _strings(value: Any, where: str) -> tuple[str, ...]:
    items = typed(value, list, where)
    if not items or not all(isinstance(v, str) and v for v in items):
        raise ConfigError(f"בהגדרות: '{where}' צריך להיות רשימת מחרוזות לא ריקה")
    return tuple(items)


def _row(raw: Any, index: int, words: dict[str, str]) -> BenchRow:
    where = f"rows[{index}]"
    table = dict(typed(raw, dict, where))
    extra = {k: table.pop(k) for k in list(table) if k in _ROW_EXTRA}
    for key in ("header_bytes", "languages", "prior"):
        if key not in extra:
            raise ConfigError(f"בהגדרות: חסר שדה ב-'{where}': {key}")
    needs = str(typed(extra.get("needs", ""), str, f"{where}.needs"))
    if needs and needs not in NEEDS:
        raise ConfigError(f"בהגדרות: '{where}.needs' לא מוכר: {needs}")
    tool = tool_row(_filled(table, words), index)
    header = int(typed(extra["header_bytes"], int, f"{where}.header_bytes"))
    if header < 0:
        raise ConfigError(f"בהגדרות: '{where}.header_bytes' שלילי")
    mode = str(typed(extra.get("hebzip_mode", ""), str, f"{where}.hebzip_mode"))
    if tool.ours != bool(mode):
        raise ConfigError(f"בהגדרות: '{where}': לשורה שלנו (ours) צריך hebzip_mode, ולאחרת - אסור")
    return BenchRow(tool, header, _strings(extra["languages"], f"{where}.languages"),
                    str(typed(extra["prior"], str, f"{where}.prior")), needs, mode,
                    int(typed(extra.get("mode_id", 0), int, f"{where}.mode_id")))


def _filled(value: Any, words: dict[str, str]) -> Any:
    """``value`` with the bench's own words (``{bench}``: the bench folder) filled in, at any depth."""
    if isinstance(value, str):
        for word, where in words.items():
            value = value.replace(word, where)
        return value
    if isinstance(value, list):
        return [_filled(v, words) for v in value]
    if isinstance(value, dict):
        return {k: _filled(v, words) for k, v in value.items()}
    return value


def parse(data: dict[str, Any], words: dict[str, str]) -> BenchSettings:
    """Validate a parsed ``bench.toml``. ``words``: placeholders known only at run time (``{paq8px}``)."""
    top = keys(data, _SECTIONS, "קובץ")
    bench = keys(top["bench"], _BENCH, "bench")
    share = float(typed(bench["seen_share"], float, "bench.seen_share"))
    if not 0 < share < 1:
        raise ConfigError("בהגדרות: bench.seen_share חייב להיות בין 0 ל-1")
    rows = tuple(_row(raw, i, words) for i, raw in enumerate(typed(top["rows"], list, "rows")))
    ids = [r.tool.id for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ConfigError("בהגדרות: רשימת השורות ריקה או שיש בה id כפול")
    return BenchSettings(_path(bench["manifest"], "bench.manifest"), _path(bench["results"], "bench.results"),
                         _path(bench["work"], "bench.work"), share,
                         _strings(bench["languages"], "bench.languages"),
                         rows, service_rows(top["services"]))


def load(path: Path, words: dict[str, str]) -> BenchSettings:
    """Read and validate a bench config file."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"לא הצלחתי לקרוא את קובץ ההגדרות: {type(error).__name__}") from error
    return parse(data, words)
