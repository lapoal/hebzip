"""From the measured pairs to the published numbers: per set and total, %, headers, timings, prior knowledge.

Rules of the table:
* a text that was in a tool's training corpus ("seen", ``overlap.json``) is never in the headline -
  it is reported in its own "seen in training" group;
* a total is shown only when the tool measured EVERY text of the group exactly (decompressed and
  compared byte for byte); anything else is "-" - never a partial sum;
* "% vs ours" compares with our best mode of that language on that group (the smallest total);
* the same totals without each tool's fixed per-package header (``header_bytes``), for the reader who
  wants the coding alone;
* timings are medians per short text, on the machine named in the file.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from atar.bench.sets import Manifest


@dataclass(frozen=True, slots=True)
class Group:
    """Texts reported together: one set, clean or seen."""

    key: str
    language: str
    length: str
    names: tuple[str, ...]


def seen_names(overlap: dict[str, Any], share: float) -> set[str]:
    """Texts that were (or may have been) in the training corpus of their language's tables."""
    return {name for name, o in overlap.get("texts", {}).items()
            if o.get("title_in_corpus") or float(o.get("shared_share", 0.0)) >= share}


def groups(manifest: Manifest, seen: set[str], measured: set[str]) -> list[Group]:
    """Every set split into its clean part and (if any) its seen part, plus one "all clean" group per
    language. Only texts that were measured at all count (a set whose dataset was not given is absent)."""
    out: list[Group] = []
    for s in manifest.sets:
        clean = tuple(t.name for t in s.texts if t.name not in seen and t.name in measured)
        dirty = tuple(t.name for t in s.texts if t.name in seen and t.name in measured)
        if clean:
            out.append(Group(s.id, s.language, s.length, clean))
        if dirty:
            out.append(Group(f"{s.id} (seen in training)", s.language, s.length, dirty))
    for language in dict.fromkeys(s.language for s in manifest.sets):
        clean = tuple(t.name for s in manifest.sets if s.language == language for t in s.texts
                      if t.name not in seen and t.name in measured)
        if clean:
            out.append(Group(f"{language}: all clean texts", language, "mixed", clean))
    return out


def _total(rows: dict[tuple[str, str], dict[str, Any]], tool: str, names: tuple[str, ...]) -> int | None:
    got = [rows.get((tool, n)) for n in names]
    if any(r is None or not r["exact"] for r in got):
        return None
    return sum(int(r["bytes"]) for r in got if r is not None)


def totals(data: dict[str, Any], gs: list[Group]) -> dict[str, dict[str, dict[str, int | None]]]:
    """group -> tool -> {as_is, no_header, raw} (None = the tool did not measure every text exactly)."""
    rows = {(r["tool"], r["text"]): r for r in data["rows"]}
    raw = {r["text"]: int(r["raw"]) for r in data["rows"]}
    out: dict[str, dict[str, dict[str, int | None]]] = {}
    for g in gs:
        per: dict[str, dict[str, int | None]] = {}
        for tool, info in data["tools"].items():
            if g.language not in info["languages"]:
                continue
            t = _total(rows, tool, g.names)
            head = int(info["header_bytes"]) * len(g.names)
            per[tool] = {"as_is": t, "no_header": None if t is None else t - head,
                         "raw": sum(raw.get(n, 0) for n in g.names)}
        out[g.key] = per
    return out


def best_ours(data: dict[str, Any], per: dict[str, dict[str, int | None]], key: str) -> str | None:
    """Our mode with the smallest total on a group (``key``: as_is / no_header)."""
    ours = [(v[key], t) for t, v in per.items() if data["tools"][t]["ours"] and v[key] is not None]
    return min(ours)[1] if ours else None


def pct(value: int | None, base: int | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return round(100.0 * (value - base) / base, 1)


def timings(data: dict[str, Any], short: set[str]) -> dict[str, dict[str, float | int]]:
    """tool -> median compress / decompress seconds over its exact short texts."""
    out: dict[str, dict[str, float | int]] = {}
    for tool in data["tools"]:
        rows = [r for r in data["rows"] if r["tool"] == tool and r["exact"] and r["text"] in short]
        if rows:
            out[tool] = {"compress_s": round(statistics.median(r["compress_s"] for r in rows), 4),
                         "decompress_s": round(statistics.median(r["decompress_s"] for r in rows), 4),
                         "texts": len(rows)}
    return out


def same_bytes(data: dict[str, Any]) -> list[str]:
    """Pairs of our rows that run the same mode two ways (service / CLI) but wrote different packages."""
    by_mode: dict[tuple[str, str], set[str]] = {}
    for r in data["rows"]:
        info = data["tools"].get(r["tool"], {})
        if info.get("ours") and r["exact"]:
            by_mode.setdefault((info["hebzip_mode"] + "@" + info["version"], r["text"]), set()).add(
                r["package_sha256"])
    return sorted(f"{m} / {t}" for (m, t), shas in by_mode.items() if len(shas) > 1)


def _versus(data: dict[str, Any], per: dict[str, dict[str, int | None]]) -> tuple[str | None, dict[str, Any]]:
    """Our best mode on a group, and every tool's totals with its % against that mode."""
    tools: dict[str, Any] = {t: dict(v) for t, v in per.items()}
    best = None
    for key, pkey in (("as_is", "pct_vs_ours"), ("no_header", "pct_vs_ours_no_header")):
        mode = best_ours(data, per, key)
        best = best or mode
        base = per[mode][key] if mode else None
        for t, v in per.items():
            tools[t][pkey] = pct(v[key], base)
    return best, tools


def summary(data: dict[str, Any], manifest: Manifest, overlap: dict[str, Any],
            share: float) -> dict[str, Any]:
    """Everything the tables show, as JSON."""
    seen = seen_names(overlap, share)
    gs = groups(manifest, seen, {r["text"] for r in data["rows"]})
    tot = totals(data, gs)
    short = {t.name for s in manifest.sets if s.length == "short" for t in s.texts}
    out_groups = []
    for g in gs:
        best, tools = _versus(data, tot[g.key])
        out_groups.append({"group": g.key, "language": g.language, "length": g.length, "texts": len(g.names),
                           "names": list(g.names), "best_ours": best, "tools": tools})
    return {"groups": out_groups, "seen": sorted(seen), "timings": timings(data, short),
            "service_vs_cli_differ": same_bytes(data)}
