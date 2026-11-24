"""The one selector: config word ``kind`` -> drawer. The null plate is the default.

the one-selector rule and the plate-by-default rule: one small file decides real-or-plate for every row; a
kind nobody
registered is refused at startup (a typo must not silently become a plate), and ``null``
is the explicit way to switch a row off without deleting it.

``SPARE`` ("one row switches back"): our rows carry the options of BOTH ways to run the
tool - ``service`` (stays loaded) and ``command`` (a new process per call) - so the user switches by
changing the one ``kind`` word. The options the chosen drawer does not own but its twin does are set
aside here; every other unknown option is still refused by the drawer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atar.compressors import command, service
from atar.compressors.command import CommandCompressor
from atar.compressors.contract import Compressor, NullCompressor
from atar.compressors.loaded import ServicePool
from atar.compressors.ppmd import Ppmd7Compressor
from atar.compressors.service import ServiceCompressor
from atar.compressors.stdlib import Bz2Compressor, GzipCompressor, XzCompressor
from atar.compressors.zstd import ZstdCompressor, ZstdDictCompressor
from atar.config.load import ToolRow
from atar.core.errors import ConfigError

Factory = Callable[[dict[str, Any], float, ServicePool], Compressor]

KINDS: dict[str, Factory] = {
    "gzip": lambda o, t, p: GzipCompressor(o),
    "xz": lambda o, t, p: XzCompressor(o),
    "bz2": lambda o, t, p: Bz2Compressor(o),
    "ppmd7": lambda o, t, p: Ppmd7Compressor(o),
    "zstd": lambda o, t, p: ZstdCompressor(o),
    "zstd_dict": lambda o, t, p: ZstdDictCompressor(o),
    "command": lambda o, t, p: CommandCompressor(o, t),
    "service": lambda o, t, p: ServiceCompressor(o, t, p),
    "null": lambda o, t, p: NullCompressor(o),
}

SPARE: dict[str, frozenset[str]] = {
    "command": frozenset(service.OPTIONS) - set(command.OPTIONS),
    "service": frozenset(command.OPTIONS) - set(service.OPTIONS),
}
"""kind -> options of its twin that may stay in the row unused (see the module text)."""


def build(row: ToolRow, pool: ServicePool | None = None) -> Compressor:
    """The drawer for one configured row (``pool``: the configured services; none = no service rows)."""
    factory = KINDS.get(row.kind)
    if factory is None:
        raise ConfigError(f"בהגדרות: סוג כלי לא מוכר '{row.kind}' (בשורה {row.id})")
    spare = SPARE.get(row.kind, frozenset())
    options = {key: value for key, value in row.options.items() if key not in spare}
    return factory(options, row.timeout_s, pool if pool is not None else ServicePool(()))
