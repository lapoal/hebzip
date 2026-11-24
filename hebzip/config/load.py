"""Load hebzip.toml: ``tomllib`` + ``core.shapes`` + the row checks.

A strict loader on ``tomllib``: a missing file or
a bad row is an error at start-up, never a silent default, and the loader never writes the user's
file. ``load_dict`` exists so tests can mutate a parsed config without a TOML writer.
Project root = the folder that holds the ``hebzip`` package. Relative table paths and the
``signature_cache`` are measured from it whatever folder the config file sits in (a test config in
``tests/_work`` must see the same ``tables/``).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from hebzip.config.schema import Settings
from hebzip.config.validate import validate
from hebzip.core.errors import ConfigError
from hebzip.core.shapes import ShapeError, build

MAX_SHOWN = 6
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "hebzip.toml"


def project_root(config_path: Path | None = None) -> Path:
    """The folder that relative paths in the config are measured from (see the module text)."""
    return PROJECT_ROOT


def read_toml(path: Path) -> dict[str, object]:
    """The parsed TOML; missing / unreadable / invalid -> ``ConfigError`` (Hebrew)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ConfigError(f"קובץ ההגדרות לא נמצא ({path})") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"קובץ ההגדרות לא נקרא ({path}): {error}") from error
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path.name}: אינו TOML תקין: {error}") from error


def load_dict(data: object, source: str = "hebzip.toml") -> Settings:
    """Shape + validate an already-parsed config -> ``Settings``."""
    try:
        settings = build(Settings, data)
    except ShapeError as error:
        shown = "; ".join(p.render() for p in error.problems[:MAX_SHOWN])
        raise ConfigError(f"{source}: {shown}") from error
    validate(settings, source, PROJECT_ROOT)
    return settings


def load_config(path: Path | None = None) -> Settings:
    """Read, shape-check and validate the config file (default: ``<project>/config/hebzip.toml``)."""
    target = path or DEFAULT_CONFIG
    return load_dict(read_toml(target), f"config\\{target.name}")
