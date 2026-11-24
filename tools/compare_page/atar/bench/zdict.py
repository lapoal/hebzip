"""Train the zstd dictionaries of the "zstd -19 + trained dictionary" rows from the corpus our tables used.

    python -m atar.bench dicts --corpus he=hewiki_random.jsonl --corpus en=enwiki_random.jsonl

Why: hebzip's numbers come with tables built from a Wikipedia sample; plain zstd starts empty. The fair
baseline hands zstd prior knowledge from the SAME sample, by zstd's own method (``zstd --train``), so the
table can say how much of our gain is "knowing Wikipedia" and how much is the model.

What it reads and how (deterministic - the same corpus file gives the same dictionary bytes):
* the corpus is one JSON article per line (``title`` + ``text``), read by ``overlap.articles``;
* every 40th line is held out exactly as the table recipes held it out (``overlap.HELD_EVERY``), so the
  dictionary sees what the tables saw and nothing more;
* one sample per non-blank line (paragraph) of every training article, in corpus order - the texts it
  will compress are short and compressed alone, and zstd's guide asks for samples that look like the data;
* zstd's ``--train`` defaults (fastCover, d=8, 4 steps over k, one thread, split 0.75, id from the
  content), at the level the rows compress with - every number is a row of ``[dicts]`` in bench.toml.

Output: ``<folder>/<language>_<size>.zdict`` and ``dicts.json`` (per file: size, sha256, dictionary id,
the corpus file's NAME and sha256, counts; the zstandard / libzstd version) - numbers and names only,
never a text and never a machine path. The dictionaries are derived from Wikipedia: CC BY-SA 4.0.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import zstandard

from atar.bench import overlap, results
from atar.bench.config import DictSettings
from atar.compressors.zstd import library
from atar.core.errors import ConfigError

RECORD = "dicts.json"


def samples(corpus: Path, held_every: int = overlap.HELD_EVERY) -> tuple[list[bytes], dict[str, int]]:
    """The training samples in corpus order, and how many articles were used / held out."""
    out: list[bytes] = []
    used = held = 0
    for i, _title, text in overlap.articles(corpus):
        if i % held_every == held_every - 1:
            held += 1
            continue
        used += 1
        out.extend(line.encode("utf-8") for line in text.split("\n") if line.strip())
    return out, {"articles": used, "held_out": held, "samples": len(out),
                 "sample_bytes": sum(map(len, out))}


def train(items: list[bytes], size: int, cfg: DictSettings) -> zstandard.ZstdCompressionDict:
    """One dictionary of ``size`` bytes (``ConfigError`` when zstd cannot train one from these samples)."""
    buffers: list[bytes | bytearray | memoryview] = list(items)
    try:
        return zstandard.train_dictionary(size, buffers, d=cfg.d, f=cfg.f, steps=cfg.steps, accel=cfg.accel,
                                          split_point=cfg.split_point, level=cfg.level, threads=0)
    except zstandard.ZstdError as error:
        raise ConfigError(f"zstd could not train a {size}-byte dictionary: {error}") from error


def file_name(language: str, size: int) -> str:
    return f"{language}_{size}.zdict"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(language: str, corpus: Path, cfg: DictSettings) -> dict[str, dict[str, Any]]:
    """Every configured size for one language: written into ``cfg.folder``; returns the file records."""
    items, counts = samples(corpus)
    corpus_info = {"language": language, "corpus": corpus.name, "corpus_sha256": _sha256(corpus), **counts}
    out: dict[str, dict[str, Any]] = {}
    cfg.folder.mkdir(parents=True, exist_ok=True)
    for size in cfg.sizes:
        made = train(items, size, cfg)
        raw = made.as_bytes()
        (cfg.folder / file_name(language, size)).write_bytes(raw)
        out[file_name(language, size)] = {**corpus_info, "asked_bytes": size, "bytes": len(raw),
                                          "sha256": hashlib.sha256(raw).hexdigest(),
                                          "dict_id": made.dict_id(), "k": made.k, "d": made.d}
    return out


def record(cfg: DictSettings, made: dict[str, dict[str, Any]]) -> Path:
    """Merge the new files' records into ``dicts.json`` (a language trained alone keeps the other's)."""
    path = cfg.folder / RECORD
    old = results.load(path) if path.is_file() else {}
    params = {k: v for k, v in asdict(cfg).items() if k not in ("folder", "sizes")}
    files = {**old.get("files", {}), **made}
    results.save(path, {"what": "zstd dictionaries for the bench (atar/bench/zdict.py)",
                        "library": library(), "params": {**params, "threads": 1},
                        "held_every": overlap.HELD_EVERY,
                        "sample": "one per non-blank line of every training article, in corpus order",
                        "licence": "CC BY-SA 4.0 (derived from Wikipedia; see LICENSE.txt here)",
                        "files": dict(sorted(files.items()))})
    return path
