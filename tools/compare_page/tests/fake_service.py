"""A fake loaded service for the lifecycle tests: hebzip's protocol, copy instead of compress.

Run as a real child process (the child path is tested with a fake child in a REAL process).
Speaks exactly the protocol of ``hebzip/interface/serve.py`` (repository root): one ready line, then one
JSON reply per JSON request line, until stdin closes. Behaviour switches (argv):
  --load-s S      sleep S seconds before the ready line
  --fail-load     say ``{"ready": false, ...}`` and exit 2
  --die-after N   exit (no reply) when request number N+1 arrives
  --hang-on OP    never answer requests whose op is OP
  --memory-on OP  answer OP with ``error_type = MemoryError``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--load-s", type=float, default=0.0)
    p.add_argument("--fail-load", action="store_true")
    p.add_argument("--die-after", type=int, default=-1)
    p.add_argument("--hang-on", default="")
    p.add_argument("--memory-on", default="")
    return p.parse_args()


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _answer(request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rid, op = request.get("id"), request.get("op")
    if op == args.memory_on:
        return {"id": rid, "ok": False, "exit": 1, "error": "x", "error_type": "MemoryError"}
    if op == "ping":
        return {"id": rid, "ok": True, "exit": 0}
    src, dst = Path(str(request.get("input"))), Path(str(request.get("output")))
    if not src.is_file():
        return {"id": rid, "ok": False, "exit": 7, "error": "missing", "error_type": "InputError"}
    shutil.copyfile(src, dst)
    return {"id": rid, "ok": True, "exit": 0, "before": src.stat().st_size, "after": dst.stat().st_size}


def main() -> int:
    args = _args()
    time.sleep(args.load_s)
    if args.fail_load:
        _send({"ready": False, "exit": 2, "error": "fake refused to load"})
        return 2
    _send({"ready": True, "modes": ["fake"], "load_s": args.load_s, "pid": os.getpid()})
    for count, line in enumerate(iter(sys.stdin.buffer.readline, b"")):
        request = json.loads(line)
        if count == args.die_after:
            return 3
        if request.get("op") == args.hang_on:
            time.sleep(3600)
        _send(_answer(request, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
