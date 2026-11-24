"""zstd with a trained dictionary: the training is deterministic and honours the held-out rule; the drawer
round-trips through its dictionary; the shipped dictionaries are the recorded ones.

The corpus here is synthetic, written by the test (never the real Wikipedia sample): numbered "articles" of
words drawn from a fixed list with a fixed seed, one JSON object per line like the real corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from atar.bench import zdict
from atar.bench.config import DEFAULT_BENCH, load
from atar.compressors.zstd import ZstdCompressor, ZstdDictCompressor
from atar.config.load import ROOT
from atar.core.errors import ToolFailed
from tests.support import WORK

WORDS = ("שלום", "עולם", "ספר", "בית", "ילד", "מים", "אור", "דרך", "עיר", "שמש", "ים", "הר", "יום", "לילה",
         "גשם", "עץ", "פרח", "שיר", "לחם", "חלב")
HELD_MARK = "סימןמוחזק"
SMALL = replace(load(DEFAULT_BENCH, {}).dicts, folder=WORK / "dicts", sizes=(4096,))


def _sentence(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))) + "."


def _corpus(name: str = "corpus.jsonl", articles: int = 160) -> Path:
    """Synthetic articles; every held-out line (index % 40 == 39) carries a mark no training line has."""
    rng = random.Random(7)
    lines = []
    for i in range(articles):
        paragraphs = [" ".join(_sentence(rng) for _ in range(3)) for _ in range(4)]
        if i % 40 == 39:
            paragraphs.append(HELD_MARK)
        lines.append(json.dumps({"title": f"t{i}", "text": "\n".join(paragraphs)}, ensure_ascii=False))
    path = WORK / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_samples_leave_out_the_held_out_lines_like_the_tables_did() -> None:
    items, counts = zdict.samples(_corpus())
    assert counts["held_out"] == 4 and counts["articles"] == 156
    assert counts["samples"] == 156 * 4 == len(items)
    assert not any(HELD_MARK.encode("utf-8") in s for s in items), "a held-out line reached the training"


def test_training_is_deterministic_same_corpus_same_dictionary() -> None:
    corpus = _corpus()
    first = zdict.build("he", corpus, SMALL)
    raw = (SMALL.folder / "he_4096.zdict").read_bytes()
    again = zdict.build("he", corpus, replace(SMALL, folder=WORK / "again"))
    assert first["he_4096.zdict"]["sha256"] == again["he_4096.zdict"]["sha256"]
    assert (WORK / "again" / "he_4096.zdict").read_bytes() == raw
    assert first["he_4096.zdict"]["sha256"] == hashlib.sha256(raw).hexdigest()
    record = json.loads(zdict.record(SMALL, first).read_text(encoding="utf-8"))
    assert record["files"]["he_4096.zdict"]["corpus"] == "corpus.jsonl", "only the file name, never a path"
    assert str(WORK) not in json.dumps(record) and WORK.as_posix() not in json.dumps(record)


def _dict_tool(corpus: Path) -> ZstdDictCompressor:
    zdict.build("he", corpus, SMALL)
    return ZstdDictCompressor({"level": 19, "dictionary": str(SMALL.folder / "he_4096.zdict")})


def test_dictionary_drawer_roundtrip_and_the_dictionary_helps() -> None:
    tool = _dict_tool(_corpus())
    src, pkg, out, plain = WORK / "t.txt", WORK / "p.bin", WORK / "r.txt", WORK / "plain.bin"
    src.write_bytes(_sentence(random.Random(99)).encode("utf-8"))
    tool.prepare()
    tool.compress(src, pkg, WORK)
    tool.decompress(pkg, out, WORK)
    assert out.read_bytes() == src.read_bytes()
    ZstdCompressor({"level": 19}).compress(src, plain, WORK)
    assert pkg.stat().st_size < plain.stat().st_size, "a trained dictionary must help on text like its corpus"
    with pytest.raises(ToolFailed):
        ZstdCompressor({"level": 19}).decompress(pkg, out, WORK)
    assert "trained dictionary 4,096 B" in tool.version() and str(WORK) not in tool.version()


def test_a_missing_dictionary_fails_the_row_never_claims_a_size() -> None:
    tool = ZstdDictCompressor({"level": 19, "dictionary": str(WORK / "nothing.zdict")})
    assert "dictionary missing" in tool.version()
    with pytest.raises(ToolFailed):
        tool.prepare()


def test_the_shipped_dictionaries_are_the_recorded_ones_and_each_row_uses_its_language() -> None:
    settings = load(DEFAULT_BENCH, {})
    assert settings.dicts.sizes == (112640, 1048576)
    record = json.loads((settings.dicts.folder / zdict.RECORD).read_text(encoding="utf-8"))
    rows = [r for r in settings.rows if r.tool.kind == "zstd_dict"]
    assert len(rows) == 4
    for row in rows:
        path = Path(row.tool.options["dictionary"])
        info = record["files"][path.name]
        assert path.parent == settings.dicts.folder and row.languages == (info["language"],)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        assert path.stat().st_size == info["bytes"] <= info["asked_bytes"]
        assert info["held_out"] > 0 and info["corpus"].endswith(".jsonl")
    assert str(ROOT.parents[1]) not in json.dumps(record)
