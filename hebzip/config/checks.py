"""The three judging helpers every validator uses - one message format for every bad row."""

from __future__ import annotations

from hebzip.core.errors import ConfigError


def bad(source: str, where: str, value: object, why: str) -> ConfigError:
    """``<file>: [section].row = value: why``."""
    shown = f'"{value}"' if isinstance(value, str) else repr(value)
    return ConfigError(f"{source}: {where} = {shown}: {why}")


def one_of(source: str, where: str, value: str, allowed: tuple[str, ...]) -> None:
    """The value must be one of the allowed words."""
    if value not in allowed:
        raise bad(source, where, value, "not one of " + ", ".join(allowed))


def sub(at: str, name: str) -> str:
    """``[modes.fast]`` + ``template`` -> ``[modes.fast.template]`` (the TOML section the user sees)."""
    return f"{at[:-1]}.{name}]"


def in_range(source: str, where: str, value: float, low: float, high: float) -> None:
    """low <= value <= high (and a bool is never a number)."""
    if isinstance(value, bool) or not low <= value <= high:
        raise bad(source, where, value, f"must be {low}..{high}")
