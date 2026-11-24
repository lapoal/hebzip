"""The summary as plain Markdown tables (``bench.md``) - what gets pasted into the README.

Why a separate writer: the JSON is the record (every number, for anyone to recompute); the Markdown is
one way to show it. Each table says what its numbers are in its heading, and the prior-knowledge table
sits right under the size tables - the reader sees what each tool brought with it before reading who won.
"""

from __future__ import annotations

from typing import Any

PCT_KEY = {"as_is": "pct_vs_ours", "no_header": "pct_vs_ours_no_header"}
WHAT = {"as_is": "package bytes as written",
        "no_header": "package bytes minus each tool's fixed per-package header"}


def _cell(value: int | float | None, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+.1f}{unit}"
    return f"{value:,}{unit}"


def _raw(group: dict[str, Any]) -> int | None:
    first: dict[str, Any] = next(iter(group["tools"].values()), {})
    raw = first.get("raw")
    return int(raw) if raw is not None else None


def _sizes(summary: dict[str, Any], tools: dict[str, Any], language: str, key: str) -> list[str]:
    groups = [g for g in summary["groups"] if g["language"] == language]
    if not groups:
        return []
    lines = ["| tool | " + " | ".join(f"{g['group']} ({g['texts']})" for g in groups) + " |",
             "|---|" + "---|" * len(groups),
             "| original text | " + " | ".join(_cell(_raw(g)) for g in groups) + " |"]
    for tool, info in tools.items():
        if language not in info["languages"]:
            continue
        cells = []
        for g in groups:
            got: dict[str, Any] = g["tools"].get(tool, {})
            cells.append(f"{_cell(got.get(key))} ({_cell(got.get(PCT_KEY[key]), '%')})")
        lines.append(f"| {info['label']} | " + " | ".join(cells) + " |")
    return lines


def _prior(tools: dict[str, Any], sizes: dict[str, int]) -> list[str]:
    lines = ["| tool | version | fixed header per package | prior knowledge shipped with the tool |",
             "|---|---|---|---|"]
    for tool, info in tools.items():
        prior = info["prior"]
        if info["ours"] and tool in sizes:
            prior = f"{sizes[tool] / 1e6:.1f} MB of tables ({prior})"
        lines.append(f"| {info['label']} | {info['version']} | {info['header_bytes']} B | {prior} |")
    return lines


def _timings(summary: dict[str, Any], tools: dict[str, Any]) -> list[str]:
    lines = ["| tool | texts | compress (median) | decompress (median) |", "|---|---|---|---|"]
    for tool, t in summary["timings"].items():
        lines.append(f"| {tools[tool]['label']} | {t['texts']} | {t['compress_s']:.3f} s "
                     f"| {t['decompress_s']:.3f} s |")
    return lines


def render(summary: dict[str, Any], data: dict[str, Any], sizes: dict[str, int]) -> str:
    """The whole Markdown page."""
    tools = data["tools"]
    out = ["# hebzip - public measurement", "", f"Machine: {data.get('machine', '?')}", ""]
    for language in dict.fromkeys(g["language"] for g in summary["groups"]):
        for key in ("as_is", "no_header"):
            out += [f"## {language}: {WHAT[key]} (in brackets: % vs our best mode on that group)", ""]
            out += [*_sizes(summary, tools, language, key), ""]
    out += ["## Prior knowledge each tool brings", "", *_prior(tools, sizes), ""]
    out += ["## Time per short text (this laptop; see the machine line)", "", *_timings(summary, tools), ""]
    seen = ", ".join(summary["seen"]) or "none"
    out += [f"Seen in training (reported apart, never in the headline groups): {seen}", ""]
    differ = summary["service_vs_cli_differ"]
    out += [f"Loaded service and cold CLI wrote the same package for every text: {differ or 'yes'}", ""]
    return "\n".join(out)
