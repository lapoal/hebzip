"""Rebuild the everyday texts from the public datasets the reader downloads (the text is not shipped).

Why rebuilt and not copied: the OMILab comments are real people's public posts (MIT-licensed
benchmark) - the repository carries only WHERE each text is (file + line) and its sha256. The rule below
is exactly the one the earlier sample was made with, so the rebuilt bytes match the pinned sha256:

* omilab - ``<dir>/<file>`` read as UTF-8 text (universal newlines), split on ``\\n``; the line's
  label column (after the last tab) is cut off, the line stripped, the benchmark's tokenisation
  spaces undone by the three ``DETOK`` rules, stripped again.

Material: the rebuilt bytes go straight to the measurement's work folder, which is deleted at the end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from atar.bench.sets import TextRef
from atar.core.errors import ConfigError

DETOK = ((re.compile(r" ([.,!?:;)\]])"), r"\1"), (re.compile(r"([(\[]) "), r"\1"),
         (re.compile(r'" (.*?) "'), r'"\1"'))
"""The benchmark is space-tokenised ("מילה ." ); these undo it (approximately, as the sample did)."""


def omilab_line(raw: str) -> str:
    """One benchmark line -> the comment as the sample holds it."""
    text = raw.rsplit("\t", 1)[0].strip()
    for pattern, repl in DETOK:
        text = pattern.sub(repl, text)
    return text.strip()


@dataclass
class Rebuilder:
    """The dataset locations given on the command line; files are read once, lazily."""

    omilab: Path | None = None
    _omilab: dict[str, list[str]] = field(default_factory=dict)

    def has(self, dataset: str) -> bool:
        """Was this dataset given?"""
        return {"omilab": self.omilab}.get(dataset) is not None

    def text(self, ref: TextRef) -> bytes:
        """The text of one dataset row."""
        if ref.dataset == "omilab" and self.omilab is not None:
            return self._omilab_text(ref).encode("utf-8")
        raise ConfigError(f"{ref.name}: the dataset '{ref.dataset}' was not given")

    def _omilab_text(self, ref: TextRef) -> str:
        name = str(ref.row["file"])
        if name not in self._omilab:
            assert self.omilab is not None
            path = self.omilab / name
            if not path.is_file():
                raise ConfigError(f"{ref.name}: {name} is not in the OMILab folder")
            self._omilab[name] = path.read_text(encoding="utf-8").split("\n")
        return omilab_line(self._omilab[name][int(ref.row["line"])])
