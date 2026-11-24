"""Was a test text in the corpus our tables were built from? Title match + shared 8-word runs.

    python -m atar.bench overlap --corpus he=<hewiki_random.jsonl> --corpus en=<enwiki_random.jsonl>

Why: a text the tables were trained on compresses unfairly well, so it
must never be in the headline numbers. Two signs are checked against every article of the corpus
(one JSON object per line, ``title`` + ``text``):
* the text's source title (from the manifest) is an article title of the corpus;
* the share of the text's 8-word runs ("shingles", words split on white space, the same definition
  the corpus builder used to drop test texts) that also occur in some article.
The table recipes skip every 40th corpus line (``held_every = 40``: held out, never counted), so an
article is reported as TRAINING or HELD-OUT by the same rule (line index % 40 == 39 -> held out);
only training articles make a text "seen". The output (``bench/overlap.json``) holds numbers, the
corpus article titles that matched (public Wikipedia titles) - never any text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atar.bench.datasets import Rebuilder
from atar.bench.sets import Manifest, text_bytes

WINDOW = 8
HELD_EVERY = 40


def shingles(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i + WINDOW]) for i in range(len(words) - WINDOW + 1)}


def articles(path: Path) -> Iterator[tuple[int, str, str]]:
    """(line index, title, text) of every article - line numbers count blank lines too."""
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if line.strip():
                row = json.loads(line)
                yield i, str(row.get("title", "")), str(row.get("text", ""))


@dataclass
class Probe:
    """One test text and what the corpus shares with it."""

    title: str
    grams: set[str]
    train: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    title_train: bool = False
    title_held: bool = False
    best: tuple[int, str] = (0, "")

    def look(self, held: bool, title: str, grams: set[str]) -> None:
        common = self.grams & grams
        if self.title and title == self.title:
            self.title_held, self.title_train = self.title_held or held, self.title_train or not held
        if common:
            (self.held if held else self.train).update(common)
            if not held and len(common) > self.best[0]:
                self.best = (len(common), title)

    def as_json(self, corpus: str) -> dict[str, Any]:
        n = len(self.grams)
        return {"corpus": corpus, "shingles": n, "title_in_corpus": self.title_train,
                "title_in_held_out": self.title_held, "shared_train": len(self.train),
                "shared_share": round(len(self.train) / n, 4) if n else 0.0,
                "shared_held_out": len(self.held),
                "best_article": self.best[1], "best_article_shared": self.best[0]}


def _probes(manifest: Manifest, language: str, rebuild: Rebuilder) -> dict[str, Probe]:
    out: dict[str, Probe] = {}
    for s in manifest.sets:
        if s.language != language or (s.dataset and not rebuild.has(s.dataset)):
            continue
        for ref in s.texts:
            text = text_bytes(ref, rebuild.text).decode("utf-8")
            out[ref.name] = Probe(ref.title, shingles(text))
    return out


def check(manifest: Manifest, language: str, corpus: Path, rebuild: Rebuilder) -> dict[str, dict[str, Any]]:
    """Every text of ``language`` against one corpus file."""
    probes = _probes(manifest, language, rebuild)
    wanted = set().union(*(p.grams for p in probes.values())) if probes else set()
    for i, title, text in articles(corpus):
        grams = shingles(text) & wanted
        held = i % HELD_EVERY == HELD_EVERY - 1
        for probe in probes.values():
            probe.look(held, title, grams)
    return {name: p.as_json(corpus.name) for name, p in probes.items()}
