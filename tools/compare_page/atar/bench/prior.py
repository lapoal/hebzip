"""The prior knowledge each tool ships with, stated next to the table (honesty about prior knowledge).

Why measured and not typed in: our modes read pinned tables (``config/hebzip.toml``); which tables a
mode reads is derived by hebzip itself (``hebzip.config.tables.mode_tables``), so the number here is
the sum of the ``bytes`` rows of exactly those tables - asked from hebzip in a child process (this
page never imports hebzip: the page and the tool stay two programs). For the other tools the row's
``prior`` text says it (brotli's built-in dictionary, paq8px's English files - their sizes are read
from the files given with ``--paq8px-dict``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from atar.config.load import REPO
from atar.core.errors import ConfigError

_ASK = ("import json, sys\n"
        "from hebzip.config.load import load_config\n"
        "from hebzip.config.tables import mode_tables\n"
        "s = load_config()\n"
        "out = {m: sum(s.tables[t].bytes for t in mode_tables(s, m)) for m in sys.argv[1:]}\n"
        "sys.stdout.write(json.dumps(out))\n")
TIMEOUT_S = 300
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def table_bytes(modes: list[str], python: str = sys.executable, repo: Path = REPO) -> dict[str, int]:
    """Mode -> bytes of the pinned tables it reads (``ConfigError`` when hebzip cannot answer)."""
    if not modes:
        return {}
    done = subprocess.run([python, "-c", _ASK, *modes], cwd=repo, capture_output=True, text=True,
                          encoding="utf-8", timeout=TIMEOUT_S, check=False, creationflags=NO_WINDOW)
    if done.returncode != 0:
        raise ConfigError(f"hebzip did not report its tables (exit {done.returncode})")
    sizes: dict[str, int] = json.loads(done.stdout)
    return sizes


def tool_table_bytes(tools: dict[str, Any]) -> dict[str, int]:
    """Our row id -> bytes of its mode's tables, each asked from the hebzip copy the row runs."""
    by_copy: dict[tuple[str, str], list[str]] = {}
    for tool, info in tools.items():
        if info["ours"]:
            by_copy.setdefault((info["python"] or sys.executable, info["repo"] or str(REPO)), []).append(tool)
    out: dict[str, int] = {}
    for (python, repo), ids in by_copy.items():
        sizes = table_bytes(sorted({tools[t]["hebzip_mode"] for t in ids}), python, Path(repo))
        out.update({t: sizes[tools[t]["hebzip_mode"]] for t in ids})
    return out


def dict_files(folder: Path | None, names: tuple[str, ...]) -> dict[str, int]:
    """Name -> size of each extra file a tool reads (empty when the folder was not given)."""
    if folder is None:
        return {}
    return {n: (folder / n).stat().st_size for n in names if (folder / n).is_file()}
