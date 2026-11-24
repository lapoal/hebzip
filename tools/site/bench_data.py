"""The recorded benchmark as the website shows it: numbers only, from ``atar.bench``'s summary + results.

Why a separate, pure step (no I/O): the page must show exactly what the bench measured - the same totals,
the same % against our best mode, the prior knowledge of every tool and its version/flags - and never
re-compute or round differently. Our "cold CLI" rows wrote the same bytes as the loaded ones (the bench
checks it), so they are folded into the loaded row as a second timing instead of a duplicate size row.
"""

from __future__ import annotations

import re
from typing import Any

COLD = "_cli"
"""Suffix of our cold-start rows in config/bench.toml (``fast_cli`` = ``fast`` run as a new process)."""


def _date(machine: str) -> str:
    found = re.search(r"measured (\d{4}-\d{2}-\d{2})", machine)
    return found.group(1) if found else ""


def _row(tool: str, info: dict[str, Any], summary: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    timing = summary["timings"].get(tool, {})
    cold = summary["timings"].get(tool + COLD, {})
    return {"id": tool, "label": info["label"], "version": info["version"], "ours": info["ours"],
            "header_bytes": info["header_bytes"], "prior": info["prior"],
            "prior_bytes": summary.get("prior_table_bytes", {}).get(tool),
            "sizes": {g["group"]: g["tools"].get(tool, {}).get("as_is") for g in groups},
            "pct": {g["group"]: g["tools"].get(tool, {}).get("pct_vs_ours") for g in groups},
            "compress_s": timing.get("compress_s"), "decompress_s": timing.get("decompress_s"),
            "cold_compress_s": cold.get("compress_s"), "cold_decompress_s": cold.get("decompress_s")}


def _raw(group: dict[str, Any]) -> int | None:
    """The original bytes of a group (every tool row carries the same ``raw``)."""
    first: dict[str, Any] = next(iter(group["tools"].values()), {})
    raw = first.get("raw")
    return int(raw) if raw is not None else None


def _language(language: str, summary: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    groups = [g for g in summary["groups"] if g["language"] == language]
    tools = [t for t, info in data["tools"].items() if language in info["languages"] and not t.endswith(COLD)]
    return {"groups": [{"id": g["group"], "texts": g["texts"], "raw": _raw(g), "best_ours": g["best_ours"]}
                       for g in groups],
            "rows": [_row(t, data["tools"][t], summary, groups) for t in tools]}


def bench_for_site(summary: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """summary.json + bench.json -> the page's recorded table (no text, no path - only numbers and labels)."""
    machine = str(data.get("machine", ""))
    languages = list(dict.fromkeys(g["language"] for g in summary["groups"]))
    return {"what": "recorded by `python -m atar.bench run` (tools/compare_page) - not measured in your browser",
            "recorded": _date(machine), "machine": machine.split(", measured")[0],
            "paq8px_dict_bytes": summary.get("paq8px_dict_bytes", {}),
            "seen_in_training": summary.get("seen", []),
            "languages": {lang: _language(lang, summary, data) for lang in languages}}
