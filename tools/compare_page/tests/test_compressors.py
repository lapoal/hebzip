"""Every drawer answers the same contract; failures become row statuses, never crashes."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from atar.compressors.command import CommandCompressor
from atar.compressors.contract import Compressor, NullCompressor
from atar.compressors.loaded import ServicePool
from atar.compressors.ppmd import Ppmd7Compressor
from atar.compressors.service import ServiceCompressor
from atar.compressors.stdlib import Bz2Compressor, GzipCompressor, XzCompressor
from atar.compressors.zstd import ZstdCompressor
from atar.config.load import ServiceRow
from atar.core.errors import NotEnoughMemory, ToolFailed
from tests.support import HEBREW, WORK

DRAWERS: list[Compressor] = [
    GzipCompressor({"level": 9}), Bz2Compressor({"level": 9}),
    XzCompressor({"preset": 9, "extreme": True}), Ppmd7Compressor({"order": 8, "memory_mb": 64}),
    ZstdCompressor({"level": 19}),
]


def _files() -> tuple[Path, Path, Path]:
    src = WORK / "text.txt"
    src.write_bytes((HEBREW * 20).encode("utf-8"))
    return src, WORK / "package.bin", WORK / "restored.bin"


_COPY = [sys.executable, "-c", "import sys,shutil; shutil.copy(sys.argv[1], sys.argv[2])", "{in}", "{out}"]
COMMAND = CommandCompressor({"version": "t", "cwd": "{work}", "compress": _COPY, "decompress": _COPY,
                             "memory_mb": 0, "memory_words": []}, 30)


_POOL = ServicePool((ServiceRow("s", "v", ".", ("no-such-program",), 1.0),))
SERVICE = ServiceCompressor({"version": "t", "service": "s", "mode": "m"}, 30, _POOL)  # never started here


@pytest.mark.parametrize("tool", [*DRAWERS, COMMAND, SERVICE, NullCompressor()],
                         ids=lambda t: type(t).__name__)
def test_contract(tool: Compressor) -> None:
    assert isinstance(tool, Compressor)
    assert tool.version()
    for name in ("version", "prepare", "compress", "decompress"):
        wanted = inspect.signature(getattr(Compressor, name))
        got = inspect.signature(getattr(tool, name))
        assert list(got.parameters) == [p for p in wanted.parameters if p != "self"], (name, got)


def test_in_process_drawers_are_always_ready_and_the_plate_never_is() -> None:
    for tool in [*DRAWERS, COMMAND]:
        tool.prepare()
    with pytest.raises(ToolFailed):
        NullCompressor().prepare()


def test_command_drawer_roundtrip() -> None:
    src, pkg, out = _files()
    COMMAND.compress(src, pkg, WORK)
    COMMAND.decompress(pkg, out, WORK)
    assert out.read_bytes() == src.read_bytes()


@pytest.mark.parametrize("tool", DRAWERS, ids=lambda t: type(t).__name__)
def test_roundtrip_and_smaller(tool: Compressor) -> None:
    src, pkg, out = _files()
    tool.compress(src, pkg, WORK)
    tool.decompress(pkg, out, WORK)
    assert out.read_bytes() == src.read_bytes()
    assert pkg.stat().st_size < src.stat().st_size


def test_null_plate_never_claims_a_result() -> None:
    src, pkg, _out = _files()
    with pytest.raises(ToolFailed):
        NullCompressor().compress(src, pkg, WORK)
    assert not pkg.exists()


def test_damaged_package_is_a_tool_failure() -> None:
    _src, pkg, out = _files()
    pkg.write_bytes(b"\x00garbage")
    for tool in DRAWERS[:3]:
        with pytest.raises(ToolFailed):
            tool.decompress(pkg, out, WORK)


def _command(args: list[str], memory_mb: int = 0, headroom: int = 10**6,
             timeout: float = 30) -> CommandCompressor:
    opts = {"version": "t", "cwd": "{work}", "compress": args, "decompress": args,
            "memory_mb": memory_mb, "memory_words": ["memory"]}
    return CommandCompressor(opts, timeout, headroom_mb=lambda: headroom)


def test_memory_guard_refuses_before_starting() -> None:
    src, pkg, _ = _files()
    tool = _command([sys.executable, "-c", "raise SystemExit(0)"], memory_mb=2300, headroom=1000)
    with pytest.raises(NotEnoughMemory) as caught:
        tool.compress(src, pkg, WORK)
    assert caught.value.status == "לא רץ (אין מספיק זיכרון)"


def test_memory_word_in_output_means_not_enough_memory() -> None:
    src, pkg, _ = _files()
    tool = _command([sys.executable, "-c", "print('Not enough memory'); raise SystemExit(3)"])
    with pytest.raises(NotEnoughMemory):
        tool.compress(src, pkg, WORK)


@pytest.mark.parametrize(("args", "status"), [
    (["C:/no/such/program.exe"], "נכשל (התוכנה לא נמצאה)"),
    ([sys.executable, "-c", "raise SystemExit(5)"], "נכשל"),
    ([sys.executable, "-c", "raise SystemExit(0)"], "נכשל"),  # exit 0 but no output file
])
def test_command_failures_have_a_status(args: list[str], status: str) -> None:
    src, pkg, _ = _files()
    with pytest.raises(ToolFailed) as caught:
        _command(args).compress(src, pkg, WORK)
    assert caught.value.status == status


def test_timeout_is_a_status() -> None:
    src, pkg, _ = _files()
    tool = _command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    with pytest.raises(ToolFailed) as caught:
        tool.compress(src, pkg, WORK)
    assert caught.value.status == "נעצר (חרג מהזמן)"


def test_placeholders_and_cwd_are_filled() -> None:
    src, pkg, _ = _files()
    script = ("import os,sys,shutil; shutil.copy(sys.argv[1], sys.argv[2]); "
              "open('cwd.txt','w').write(os.getcwd())")
    _command([sys.executable, "-c", script, "{in}", "{out}"]).compress(src, pkg, WORK)
    assert pkg.read_bytes() == src.read_bytes()
    assert Path((WORK / "cwd.txt").read_text()).resolve() == WORK.resolve()
