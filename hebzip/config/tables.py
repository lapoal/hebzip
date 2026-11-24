"""Which ``[tables.*]`` a mode reads.

Why only this here: the tables are the second half of the format - a mode's tables are checked
against their sha256 rows before anything is built. The check itself lives in
the table store (``store/eager.py`` checks the pinned files, ``store/mapped.py`` the derived copy
against its manifest and the same rows), so this module only answers WHICH tables, from the
drawer rows (``config/drawers.py``). The project owns its tables (``tables/``).
"""

from __future__ import annotations

from hebzip.config.drawers import table_readers
from hebzip.config.schema import Settings
from hebzip.core.errors import ConfigError


def mode_tables(settings: Settings, mode: str) -> list[str]:
    """The names of every ``[tables.*]`` the mode reads (sorted, no duplicates) - derived from its drawer
    rows by ``config/drawers.py`` (no hand-kept list, no branch on a drawer word)."""
    if mode not in settings.modes:
        raise ConfigError(f"[modes.{mode}]: no such mode")
    return sorted(table_readers(settings, [mode]))
