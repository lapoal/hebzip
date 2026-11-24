"""The text sets of the measurement (``bench/manifest.json``): which texts, from where, pinned by sha256.

Why a manifest and not a folder glob: the numbers are published, so every text must be named with
its source (title, URL, date - or dataset and row) and must be the SAME text for whoever re-runs the
bench. A text shipped in the repository is read from its file; a text of a public dataset that we may
not redistribute is rebuilt from the dataset the reader downloads (``datasets.py``). Either way its
sha256 is checked before it is measured - a mismatch skips the text with a clear line, never a silent
substitute. A set whose dataset was not given is skipped as a whole, saying so.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atar.config.load import REPO
from atar.core.errors import ConfigError

_SET_KEYS = {"id", "language", "length", "kind", "about", "texts"}
LENGTHS = ("short", "long", "mixed")


@dataclass(frozen=True, slots=True)
class TextRef:
    """One text of a set: where it comes from and what it must be."""

    set_id: str
    name: str
    language: str
    length: str
    sha256: str
    bytes: int
    file: str
    dataset: str
    row: dict[str, Any]
    title: str = ""


@dataclass(frozen=True, slots=True)
class TextSet:
    """One set as the manifest names it."""

    id: str
    language: str
    length: str
    kind: str
    texts: tuple[TextRef, ...]

    @property
    def dataset(self) -> str:
        """The dataset its texts are rebuilt from ('' = shipped in the repository)."""
        return self.texts[0].dataset if self.texts else ""


@dataclass(frozen=True, slots=True)
class Manifest:
    sets: tuple[TextSet, ...]
    datasets: dict[str, dict[str, Any]]


def _text(raw: dict[str, Any], s: dict[str, Any]) -> TextRef:
    name = str(raw["name"])
    file, dataset = str(raw.get("file", "")), str(raw.get("dataset", ""))
    if bool(file) == bool(dataset):
        raise ConfigError(f"manifest: הטקסט {name} צריך file או dataset (אחד מהם)")
    return TextRef(str(s["id"]), name, str(s["language"]), str(s["length"]), str(raw["sha256"]),
                   int(raw["bytes"]), file, dataset, dict(raw.get("from", {})),
                   str(raw.get("source", {}).get("title") or ""))


def _set(raw: dict[str, Any]) -> TextSet:
    unknown = sorted(set(raw) - _SET_KEYS)
    if unknown or raw.get("length") not in LENGTHS:
        raise ConfigError(f"manifest: קבוצה {raw.get('id')} לא תקינה ({', '.join(unknown) or 'length'})")
    texts = tuple(_text(t, raw) for t in raw["texts"])
    if len({t.dataset for t in texts}) > 1:
        raise ConfigError(f"manifest: בקבוצה {raw['id']} מעורבים מקורות שונים")
    return TextSet(str(raw["id"]), str(raw["language"]), str(raw["length"]), str(raw["kind"]), texts)


def load_manifest(path: Path) -> Manifest:
    """Read and check the manifest (unknown set field, a text with both or neither source = refused)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"לא הצלחתי לקרוא את המניפסט: {type(error).__name__}") from error
    sets = tuple(_set(s) for s in data["sets"])
    ids = [s.id for s in sets]
    names = [t.name for s in sets for t in s.texts]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ConfigError("manifest: id של קבוצה או שם של טקסט חוזר פעמיים")
    return Manifest(sets, dict(data["datasets"]))


Rebuild = Callable[[TextRef], bytes]
"""Dataset text -> its bytes (``datasets.Rebuilder.text``)."""


def text_bytes(ref: TextRef, rebuild: Rebuild | None) -> bytes:
    """The text, checked against its sha256 (``ConfigError`` naming the text - never its content)."""
    if ref.file:
        data = (REPO / ref.file).read_bytes()
    elif rebuild is not None:
        data = rebuild(ref)
    else:
        raise ConfigError(f"{ref.name}: the dataset '{ref.dataset}' was not given")
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ConfigError(f"{ref.name}: the text is not the pinned one (sha256 differs)")
    return data
