"""A row whose tool stays loaded: our ``hebzip serve``, asked through ``LoadedService``.

Same contract as every drawer (file in, file out, ``ToolFailed`` with the row's word). ``prepare`` waits
until the service is loaded (or restarted), so the time the table shows is the request time only -
compress including its check-decode, and decompress - and never the one-time load, which the page shows
on its own line. "נפתח זהה למקור" is still earned by the runner, by decompressing and comparing bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atar.compressors.loaded import ServicePool
from atar.compressors.stdlib import check_options
from atar.core.errors import NotEnoughMemory, ToolFailed

OPTIONS: dict[str, type] = {"version": str, "service": str, "mode": str}
MEMORY_KINDS = ("MemoryError",)
"""``error_type`` words of a reply that mean "not enough memory" (the row says so, like paq8px's)."""


class ServiceCompressor:
    """One mode of a loaded service."""

    def __init__(self, options: dict[str, Any], timeout_s: float, pool: ServicePool) -> None:
        opts = check_options(options, OPTIONS, "service")
        self.service = pool.take(str(opts["service"]))
        self.mode = str(opts["mode"])
        self._version = str(opts["version"])
        self.timeout_s = timeout_s

    def version(self) -> str:
        return self._version

    def prepare(self) -> None:
        self.service.ensure_ready()

    def compress(self, src: Path, dst: Path, work: Path) -> None:
        self._ask({"op": "compress", "input": str(src), "output": str(dst), "mode": self.mode,
                   "fallback": True, "force": True}, dst)

    def decompress(self, src: Path, dst: Path, work: Path) -> None:
        self._ask({"op": "decompress", "input": str(src), "output": str(dst), "force": True}, dst)

    def _ask(self, request: dict[str, Any], dst: Path) -> None:
        reply = self.service.ask(request, self.timeout_s)
        if reply.get("ok") is not True:
            if reply.get("error_type") in MEMORY_KINDS:
                raise NotEnoughMemory()
            raise ToolFailed(f"הכלי נכשל (קוד יציאה {reply.get('exit')})")
        if not dst.is_file():
            raise ToolFailed("הכלי לא כתב קובץ פלט")
