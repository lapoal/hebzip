"""The loaded-tool drawer and its process: background start, waiting, failures, restart, stop.

A fake service (``tests/fake_service.py``) in a real child process speaks hebzip's protocol, so
every case is fast and exact; the real hebzip service is exercised in ``test_full_http.py`` and
``test_window_close.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from atar.compressors.loaded import FELL, NOT_LOADED, ServicePool
from atar.compressors.registry import build
from atar.compressors.service import ServiceCompressor
from atar.config.load import ROOT, ServiceRow, load
from atar.core.errors import ConfigError, NotEnoughMemory, ToolFailed
from atar.services.runner import run_tool
from tests.procs import alive, gone, kill_tree, tree
from tests.support import HEBREW, WORK, row

FAKE = ROOT / "tests" / "fake_service.py"


def _pool(*args: str, ready_timeout_s: float = 30, program: str = sys.executable) -> ServicePool:
    command = (program, str(FAKE), *args)
    return ServicePool((ServiceRow("fake", "fake 1", str(ROOT), command, ready_timeout_s),))


def _drawer(pool: ServicePool, timeout_s: float = 30) -> ServiceCompressor:
    return ServiceCompressor({"version": "v", "service": "fake", "mode": "m"}, timeout_s, pool)


def _files() -> tuple[Path, Path, Path]:
    src = WORK / "text.txt"
    src.write_bytes(HEBREW.encode("utf-8"))
    return src, WORK / "package.bin", WORK / "restored.bin"


@pytest.fixture
def pools() -> Any:
    made: list[ServicePool] = []
    yield made
    for pool in made:
        pool.stop_all()


def test_starts_in_the_background_and_a_request_waits_for_ready(pools: list[ServicePool]) -> None:
    pool = _pool("--load-s", "1.5")
    pools.append(pool)
    tool = _drawer(pool)
    started = time.monotonic()
    pool.start_all()
    assert time.monotonic() - started < 0.5  # the page opens at once
    assert pool.snapshot()[0]["state"] == "loading"
    src, pkg, out = _files()
    tool.prepare()  # an upload before "ready" waits instead of failing
    tool.compress(src, pkg, WORK)
    tool.decompress(pkg, out, WORK)
    assert out.read_bytes() == src.read_bytes()
    snap = pool.snapshot()[0]
    assert snap["state"] == "ready" and snap["load_s"] >= 1.5 and snap["modes"] == ["fake"]


def test_the_row_time_is_the_request_not_the_load(pools: list[ServicePool]) -> None:
    pool = _pool("--load-s", "2")
    pools.append(pool)
    src, _, _ = _files()
    (WORK / "job").mkdir()
    outcome = run_tool(row("ours", "service", {}), _drawer(pool), src, WORK / "job")
    assert outcome.status == "ok" and outcome.identical is True
    assert outcome.compress_s is not None and outcome.compress_s < 1.0, outcome


def test_a_failed_request_is_a_row_status_and_the_service_goes_on(pools: list[ServicePool]) -> None:
    pool = _pool()
    pools.append(pool)
    tool = _drawer(pool)
    src, pkg, _ = _files()
    with pytest.raises(ToolFailed) as caught:
        tool.compress(WORK / "missing.txt", pkg, WORK)
    assert "קוד יציאה 7" in str(caught.value)
    pid = pool.used["fake"].pid()
    tool.compress(src, pkg, WORK)
    assert pool.used["fake"].pid() == pid and pool.snapshot()[0]["starts"] == 1


def test_a_service_that_died_is_restarted_on_next_use(pools: list[ServicePool]) -> None:
    pool = _pool("--die-after", "1")
    pools.append(pool)
    tool = _drawer(pool)
    src, pkg, out = _files()
    tool.compress(src, pkg, WORK)
    with pytest.raises(ToolFailed) as caught:
        tool.decompress(pkg, out, WORK)  # the fake dies on its 2nd request
    assert caught.value.status == FELL
    assert pool.snapshot()[0]["state"] == "stopped"
    tool.decompress(pkg, out, WORK)  # next use: a new process
    assert out.read_bytes() == src.read_bytes() and pool.snapshot()[0]["starts"] == 2


def test_a_service_killed_while_idle_is_restarted_on_next_use(pools: list[ServicePool]) -> None:
    pool = _pool()
    pools.append(pool)
    tool = _drawer(pool)
    tool.prepare()
    pid = pool.used["fake"].pid()
    assert pid is not None
    procs = tree(pid)
    kill_tree(pid)  # e.g. Task Manager, or out of memory while nobody was asking
    assert gone(procs)
    src, pkg, out = _files()
    tool.prepare()
    tool.compress(src, pkg, WORK)
    tool.decompress(pkg, out, WORK)
    assert out.read_bytes() == src.read_bytes() and pool.snapshot()[0]["starts"] == 2


def test_a_service_that_cannot_load_says_so(pools: list[ServicePool]) -> None:
    refusing, missing = _pool("--fail-load"), _pool(program="C:/no/such/python.exe")
    pools += [refusing, missing]
    for pool, words in ((refusing, "fake refused to load"), (missing, "לא נמצאה")):
        with pytest.raises(ToolFailed) as caught:
            _drawer(pool).prepare()
        assert caught.value.status == NOT_LOADED
        snap = pool.snapshot()[0]
        assert snap["state"] == "failed" and words in snap["error"], snap


def test_a_request_over_its_budget_kills_the_process(pools: list[ServicePool]) -> None:
    pool = _pool("--hang-on", "compress")
    pools.append(pool)
    tool = _drawer(pool, timeout_s=1)
    tool.prepare()
    pid = pool.used["fake"].pid()
    assert pid is not None
    procs = tree(pid)
    src, pkg, _ = _files()
    with pytest.raises(ToolFailed) as caught:
        tool.compress(src, pkg, WORK)
    assert caught.value.status == "נעצר (חרג מהזמן)"
    assert gone(procs), [p for p in procs if alive(p)]


def test_out_of_memory_in_the_service_is_the_memory_status(pools: list[ServicePool]) -> None:
    pool = _pool("--memory-on", "compress")
    pools.append(pool)
    src, pkg, _ = _files()
    with pytest.raises(NotEnoughMemory):
        _drawer(pool).compress(src, pkg, WORK)


def test_stop_leaves_no_process() -> None:
    pool = _pool()
    tool = _drawer(pool)
    tool.prepare()
    pid = pool.used["fake"].pid()
    assert pid is not None
    procs = tree(pid)
    assert len(procs) >= 1 and all(alive(p) for p in procs)
    pool.stop_all()
    assert gone(procs), [p for p in procs if alive(p)]
    assert pool.snapshot()[0]["state"] == "stopped"


def test_one_kind_word_switches_our_rows_between_service_and_command() -> None:
    settings = load()
    ours = [r for r in settings.tools if r.ours]
    assert len(ours) == 3 and all(r.kind == "service" for r in ours)  # fast, max, max_plus
    pool = ServicePool(settings.services)
    assert all(isinstance(build(r, pool), ServiceCompressor) for r in ours)
    assert list(pool.used) == ["hebzip"]  # all our rows share ONE loaded process
    back = ServicePool(settings.services)
    switched = [type(build(row(r.id, "command", r.options, True), back)).__name__ for r in ours]
    assert switched == ["CommandCompressor"] * len(ours) and back.used == {}  # nothing to load
    with pytest.raises(ConfigError):
        build(row("x", "service", {**ours[0].options, "colour": "green"}), pool)
    with pytest.raises(ConfigError):
        build(row("x", "service", {**ours[0].options, "service": "nope"}), pool)
