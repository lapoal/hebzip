"""The loaded service (``python -m hebzip serve``) - always through the real entry point in a child.

What is proven:
* the service writes the CLI's package bytes, byte for byte, for max / max_plus / generic on golden
  texts, in both engines; and each side opens the other's package;
* no state leaks between requests: compress A, compress B, compress A again (same bytes), decode A's package
  in the same process (= A), and a fresh service decodes packages made by another process (the real-path
  rule: test what the user runs);
* a failed request is a reply and the service goes on; a mode that does not exist refuses to start;
* closing stdin ends the service, and so does the death of the parent - launcher AND real interpreter.
Texts: golden texts only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.serve_client import NO_WINDOW, Service, alive, children, kill_tree, serve_command
from tests.support import ENGINES, ROOT, TEXTS, compress_file, engine_config, work_dir

MODES = ("max", "max_plus", "generic")
PUBLIC = ("own_letter", "wiki_bicycle", "wiki_rain")
EXIT_INPUT, EXIT_CONFIG = 7, 2


def _text(name: str) -> Path:
    return TEXTS / f"{name}.txt"


@pytest.mark.parametrize("engine", ENGINES)
def test_the_service_writes_the_cli_bytes(engine: str) -> None:
    cfg = engine_config(engine, f"serve_cfg_{engine}")
    work = work_dir(f"serve_bytes_{engine}")
    with Service(*MODES, config=cfg) as service:
        assert service.ready.get("ready") is True, service.ready
        for mode in MODES:
            for name in PUBLIC:
                cli_pkg, srv_pkg, back = work / f"{mode}_{name}.cli", work / f"{mode}_{name}.srv", work / "back"
                done = compress_file(_text(name), cli_pkg, mode, "--config", str(cfg))
                assert done.returncode == 0, done.stderr[-300:]
                reply = service.compress(_text(name), srv_pkg, mode)
                assert reply["ok"] and reply["exit"] == 0, reply
                assert srv_pkg.read_bytes() == cli_pkg.read_bytes(), (engine, mode, name)
                assert service.decompress(cli_pkg, back)["ok"] and back.read_bytes() == _text(name).read_bytes()
    assert service.proc.returncode == 0


def test_no_state_leaks_between_requests() -> None:
    work = work_dir("serve_state")
    a, b = _text("wiki_bicycle"), _text("own_story")
    with Service("max", "max_plus") as service:
        for mode in ("max", "max_plus"):
            first, other, again, back = (work / f"{mode}.{s}" for s in ("a1", "b", "a2", "out"))
            assert service.compress(a, first, mode)["ok"]
            assert service.compress(b, other, mode)["ok"]
            assert service.compress(a, again, mode)["ok"]
            assert again.read_bytes() == first.read_bytes(), f"{mode}: A's package changed after B"
            assert service.decompress(first, back)["ok"] and back.read_bytes() == a.read_bytes()
            assert service.decompress(other, back)["ok"] and back.read_bytes() == b.read_bytes()
    with Service("max", "max_plus") as fresh:  # another process: only the package bytes travel
        for mode in ("max", "max_plus"):
            back = work / f"{mode}.fresh"
            assert fresh.decompress(work / f"{mode}.b", back)["ok"] and back.read_bytes() == b.read_bytes()
            assert fresh.decompress(work / f"{mode}.a1", back)["ok"] and back.read_bytes() == a.read_bytes()


def test_a_failed_request_is_a_reply_and_the_service_goes_on() -> None:
    work = work_dir("serve_fail")
    pkg, back = work / "x.hz", work / "x.out"
    with Service("max") as service:
        service.send_line(b"this is not json")
        assert service.next(60)["exit"] == EXIT_INPUT
        assert service.ask("explode")["exit"] == EXIT_INPUT
        assert service.ask("compress", input=work / "missing.txt", output=pkg, mode="max")["exit"] == EXIT_INPUT
        assert service.ask("compress", input=_text("own_letter"), output=pkg, mode="no_such_mode")["exit"] == 2
        assert service.ask("compress", input=_text("own_letter"), mode="max")["exit"] == EXIT_INPUT  # no output
        junk = work / "junk.hz"
        junk.write_bytes(b"not a package at all")
        assert service.decompress(junk, back)["exit"] == 4 and not back.exists()
        pkg.write_bytes(b"precious")
        refused = service.ask("compress", input=_text("own_letter"), output=pkg, mode="max", force=False)
        assert refused["exit"] == EXIT_INPUT and pkg.read_bytes() == b"precious"
        assert service.compress(_text("own_letter"), pkg, "max")["ok"]
        assert service.decompress(pkg, back)["ok"] and back.read_bytes() == _text("own_letter").read_bytes()
        assert service.ask("ping")["ok"]
    assert service.proc.returncode == 0


def test_an_unknown_mode_refuses_to_start() -> None:
    with Service("no_such_mode") as service:
        assert service.ready["ready"] is False and service.ready["exit"] == EXIT_CONFIG, service.ready
        assert service.proc.wait(60) == EXIT_CONFIG


def _gone(pids: list[int], budget_s: float = 30) -> bool:
    end = time.monotonic() + budget_s
    while time.monotonic() < end:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.5)
    return False


def test_closing_stdin_ends_the_service_and_its_interpreter() -> None:
    service = Service("max")
    try:
        assert service.ready["ready"] is True
        tree = [service.proc.pid, *children(service.proc.pid)]
        assert len(tree) >= 1 and all(alive(p) for p in tree)
        assert service.close(60) == 0
        assert _gone(tree), f"left running: {[p for p in tree if alive(p)]}"
    finally:
        kill_tree(service.proc.pid)


PARENT = """
import json, os, subprocess, sys, time
cmd = json.loads(sys.argv[1])
child = subprocess.Popen(cmd, cwd=sys.argv[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
ready = json.loads(child.stdout.readline())
sys.stdout.write(json.dumps({"parent": os.getpid(), "service": child.pid, "ready": ready}) + "\\n")
sys.stdout.flush()
time.sleep(600)
"""


def test_the_service_ends_when_its_parent_dies() -> None:
    """The page's window closed = the page process killed, nothing closed politely (no /T: only the parent)."""
    parent = subprocess.Popen([sys.executable, "-c", PARENT, json.dumps(serve_command("max")), str(ROOT)],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    try:
        assert parent.stdout is not None
        info = json.loads(parent.stdout.readline())
        assert info["ready"]["ready"] is True
        tree = [info["service"], *children(info["service"])]
        assert all(alive(p) for p in tree)
        subprocess.run(["taskkill", "/F", "/PID", str(info["parent"])], capture_output=True, check=False)
        assert _gone(tree), f"orphans left after the parent died: {[p for p in tree if alive(p)]}"
    finally:
        kill_tree(parent.pid)
