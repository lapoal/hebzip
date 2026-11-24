"""Check a JSON value against a typing annotation at run time - one checker for config + artifacts.

Why it exists (PLAN_MODULAR 3.2 + 4.3): every file between the stages gets a type, and every
settings file gets a typed schema where "an unknown field = an error at startup". Writing the
type once for mypy and again as a hand-made validator would let the two drift apart. So the
type *is* the schema: a ``TypedDict`` (artifacts - the dict stays the dict that is on disk, the
format cannot change) or a frozen ``dataclass`` (settings - built into an immutable object).

Why in ``core`` and not ``domain``: ``config`` may import only ``core`` (PLAN_MODULAR 1), and both
``config`` and ``domain`` need it. It is pure: no files, no clock.

Supported annotations: ``str int float bool None Any object``, ``Literal[...]``, unions
(``X | None``), ``list[X]``, ``tuple[X, ...]``, ``dict[str, X]`` / ``Mapping[str, X]``,
``NewType``, ``type`` aliases, ``TypedDict`` (``NotRequired`` keys) and dataclasses.
Rules that matter:

* ``bool`` is never accepted as a number (JSON ``true`` is not ``1``).
* ``float`` accepts an int (JSON writes ``250`` for ``250.0`` - the old files do, e.g. ``a: 0``).
* an unknown key is a problem - in a TypedDict and in a dataclass alike.
* a dataclass field may read another JSON key (``metadata={"key": "_pending"}``), and one field
  marked ``metadata={"notes": True}`` collects every other ``_``-key (``_why``, ``_about``) as
  documentation; without such a field an ``_``-key is unknown, like any other.

Problems are Hebrew (they reach the user through ``ConfigError`` / ``ArtifactError``).
"""

from __future__ import annotations

import dataclasses
import functools
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

MAX_PROBLEMS = 20
_LISTS = (list, Sequence)
_MAPS = (dict, Mapping)
_HE: dict[type, str] = {str: "טקסט", int: "מספר שלם", float: "מספר", bool: "true/false",
                        type(None): "null", list: "רשימה", dict: "אובייקט"}


@dataclass(frozen=True)
class Problem:
    """One mismatch: where in the value (``$.audio.pad_sec``) and what is wrong (Hebrew)."""

    where: str
    what: str

    def render(self) -> str:
        """``$.audio.pad_sec: צריך מספר, נמצא טקסט``."""
        return f"{self.where}: {self.what}"


class ShapeError(ValueError):
    """Raised by ``build`` - carries the problems; the caller wraps it in a typed core error."""

    def __init__(self, problems: list[Problem]) -> None:
        super().__init__("; ".join(p.render() for p in problems))
        self.problems = problems


def check(tp: object, value: object, where: str = "$") -> list[Problem]:
    """Every problem of ``value`` against ``tp`` (at most ``MAX_PROBLEMS``); ``[]`` = matches."""
    out: list[Problem] = []
    _convert(tp, value, where, out)
    return out[:MAX_PROBLEMS]


def build[T](cls: type[T], value: object, where: str = "$") -> T:
    """Build a (frozen) dataclass from a JSON value, or raise ``ShapeError`` with every problem."""
    out: list[Problem] = []
    built = _convert(cls, value, where, out)
    if out:
        raise ShapeError(out[:MAX_PROBLEMS])
    if not isinstance(built, cls):
        raise ShapeError([Problem(where, "לא נבנה אובייקט")])
    return built


def _kind(value: object) -> str:
    return _HE.get(type(value), type(value).__name__)


def _name(tp: object) -> str:
    if isinstance(tp, type):
        return _HE.get(tp, tp.__name__)
    return str(tp).replace("typing.", "")


def _convert(tp: object, value: object, where: str, out: list[Problem]) -> object:
    if len(out) >= MAX_PROBLEMS or tp is Any or tp is object:
        return value
    if isinstance(tp, TypeAliasType):
        return _convert(tp.__value__, value, where, out)
    supertype = getattr(tp, "__supertype__", None)          # typing.NewType
    if supertype is not None:
        return _convert(supertype, value, where, out)
    return _convert_shaped(tp, value, where, out)


def _convert_shaped(tp: object, value: object, where: str, out: list[Problem]) -> object:
    origin = get_origin(tp)
    if origin is not None:
        return _convert_generic(tp, origin, value, where, out)
    if typing.is_typeddict(tp):
        return _convert_typeddict(tp, value, where, out)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _convert_dataclass(tp, value, where, out)
    return _convert_plain(tp, value, where, out)


def _convert_plain(tp: object, value: object, where: str, out: list[Problem]) -> object:
    if tp is type(None) or tp is None:
        ok = value is None
    elif tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    elif tp is str:
        ok = isinstance(value, str)
    else:
        raise TypeError(f"shapes: unsupported annotation {tp!r}")
    if not ok:
        out.append(Problem(where, f"צריך {_name(tp)}, נמצא {_kind(value)}"))
    return value


def _convert_generic(tp: object, origin: object, value: object, where: str,
                     out: list[Problem]) -> object:
    args = get_args(tp)
    if origin is Literal:
        return _convert_literal(args, value, where, out)
    if origin is Union or origin is types.UnionType:
        return _convert_union(args, value, where, out)
    if origin is typing.NotRequired or origin is typing.Required:
        return _convert(args[0], value, where, out)
    if origin is tuple:
        return tuple(_convert_items(args[0], value, where, out))
    if origin in _LISTS:
        return _convert_items(args[0], value, where, out)
    if origin in _MAPS:
        return _convert_map(args[1], value, where, out)
    raise TypeError(f"shapes: unsupported annotation {tp!r}")


def _convert_literal(args: tuple[object, ...], value: object, where: str,
                     out: list[Problem]) -> object:
    if not any(type(value) is type(a) and value == a for a in args):
        allowed = " | ".join(repr(a) for a in args)
        out.append(Problem(where, f"הערך {value!r} לא מותר; מותר: {allowed}"))
    return value


def _convert_union(args: tuple[object, ...], value: object, where: str,
                   out: list[Problem]) -> object:
    tries: list[tuple[object, list[Problem]]] = []
    for member in args:
        found: list[Problem] = []
        converted = _convert(member, value, where, found)
        if not found:
            return converted
        tries.append((member, found))
    real = [(m, f) for m, f in tries if m is not type(None)]
    if value is not None and len(real) == 1:
        out.extend(real[0][1])
    else:
        names = " או ".join(_name(m) for m in args)
        out.append(Problem(where, f"צריך {names}, נמצא {_kind(value)}"))
    return value


def _convert_items(item_tp: object, value: object, where: str, out: list[Problem]) -> list[object]:
    if not isinstance(value, list):
        out.append(Problem(where, f"צריך רשימה, נמצא {_kind(value)}"))
        return []
    return [_convert(item_tp, item, f"{where}[{i}]", out) for i, item in enumerate(value)]


def _convert_map(value_tp: object, value: object, where: str,
                 out: list[Problem]) -> dict[str, object]:
    if not isinstance(value, dict):
        out.append(Problem(where, f"צריך אובייקט, נמצא {_kind(value)}"))
        return {}
    return {str(k): _convert(value_tp, v, f"{where}[{k!r}]", out) for k, v in value.items()}


@functools.cache
def _hints(tp: type) -> dict[str, Any]:
    return get_type_hints(tp)


@functools.cache
def _required_keys(tp: Any) -> frozenset[str]:
    """The keys a TypedDict really requires.

    Why not just ``__required_keys__``: on
    Python 3.12, a class-based TypedDict whose module has ``from __future__ import annotations``
    sees ``NotRequired[...]`` only as a string, so the key is counted as *required* - a status
    file without ``started`` was rejected. The resolved hints (with extras) do carry the
    ``NotRequired`` / ``Required`` wrapper, so they decide.
    """
    required = set(tp.__required_keys__)
    for key, hint in get_type_hints(tp, include_extras=True).items():
        origin = get_origin(hint)
        if origin is typing.NotRequired:
            required.discard(key)
        elif origin is typing.Required:
            required.add(key)
    return frozenset(required)


def _convert_typeddict(tp: Any, value: object, where: str, out: list[Problem]) -> object:
    if not isinstance(value, dict):
        out.append(Problem(where, f"צריך אובייקט, נמצא {_kind(value)}"))
        return value
    hints = _hints(tp)
    for key in sorted(_required_keys(tp) - value.keys()):
        out.append(Problem(where, f"חסר השדה '{key}'"))
    for key, item in value.items():
        if key not in hints:
            out.append(Problem(where, f"שדה לא מוכר '{key}'"))
        else:
            _convert(hints[key], item, f"{where}.{key}", out)
    return value


def _convert_dataclass(tp: type, value: object, where: str, out: list[Problem]) -> object:
    if not isinstance(value, dict):
        out.append(Problem(where, f"צריך אובייקט, נמצא {_kind(value)}"))
        return value
    hints = _hints(tp)
    start = len(out)
    kwargs: dict[str, object] = {}
    used: set[str] = set()
    notes_field: str | None = None
    for f in dataclasses.fields(tp):
        if f.metadata.get("notes"):
            notes_field = f.name
            continue
        key = str(f.metadata.get("key", f.name))
        used.add(key)
        if key in value:
            kwargs[f.name] = _convert(hints[f.name], value[key], f"{where}.{key}", out)
        elif not _has_default(f):
            out.append(Problem(where, f"חסר השדה '{key}'"))
    notes = _collect_notes(value, used, notes_field is not None, where, out)
    if notes_field is not None:
        kwargs[notes_field] = notes
    if len(out) > start:
        return value
    return tp(**kwargs)


def _has_default(f: dataclasses.Field[object]) -> bool:
    return f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING


def _collect_notes(value: dict[str, object], used: set[str], has_notes: bool, where: str,
                   out: list[Problem]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for key, item in value.items():
        if key in used:
            continue
        if has_notes and key.startswith("_") and isinstance(item, str):
            notes[key] = item
        else:
            out.append(Problem(where, f"שדה לא מוכר '{key}'"))
    return notes
