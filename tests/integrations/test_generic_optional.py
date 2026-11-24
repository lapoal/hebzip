"""zstandard and pyppmd are optional: imported when a codec of theirs runs, never at import (``integrations/generic``).

Why (the browser build): Pyodide has neither wheel, and the top-level imports made the whole package
fail to import there, although mode ``fast`` never touches them. Proven in a child interpreter where both
modules (and numba) are blocked: the package imports, ``fast`` compresses to the golden bytes, and the codecs
that need a missing library refuse with a typed error naming the package (never a silently shorter list).
With the libraries installed nothing changed: a damaged zstd frame is still ``CorruptPackage`` (ZstdError).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hebzip.core.errors import CorruptPackage
from hebzip.domain.varint import encode_varint
from hebzip.integrations.generic import CODECS, BestOfCodec
from tests.support import ROOT, TEXTS, golden_package

BLOCKED_CHILD = """
import sys
for name in ("zstandard", "pyppmd", "numba"):
    sys.modules[name] = None
from pathlib import Path
from hebzip.config.load import load_dict
from hebzip.core.errors import DrawerUnavailable
from hebzip.integrations.generic import CODECS, BestOfCodec
from hebzip.services.compress import compress_bytes
from hebzip.services.select import Assembly
from tests.support import python_config
cfg = python_config()
cfg["hebzip"]["fallback"] = []
cfg["hebzip"]["table_store"] = "eager"
asm = Assembly(load_dict(cfg), Path.cwd())
sys.stdout.buffer.write(compress_bytes(asm, Path(sys.argv[1]).read_bytes(), "fast").data)
for word, package in (("zstd:19", "zstandard"), ("ppmd:H:6", "pyppmd")):
    family, *options = word.split(":")
    codec = BestOfCodec([word], [CODECS[family](options, 16)])
    try:
        codec.encode(b"abc")
    except DrawerUnavailable as error:
        assert package in str(error), str(error)
    else:
        raise SystemExit(f"{word}: no refusal without {package}")
assert all(sys.modules.get(n) is None for n in ("zstandard", "pyppmd", "numba"))
"""


def test_fast_runs_and_missing_codecs_refuse_without_the_optional_libraries() -> None:
    src = TEXTS / "own_message.txt"
    done = subprocess.run([sys.executable, "-c", BLOCKED_CHILD, str(src)], cwd=ROOT, capture_output=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-1500:]
    assert done.stdout == golden_package("fast", "own_message")


def test_a_damaged_zstd_frame_is_still_a_corrupt_package() -> None:
    codec = BestOfCodec(["zstd:19"], [CODECS["zstd"](["19"], 16)])
    frame = codec.encode(b"hello hello hello")
    damaged = bytes([0]) + encode_varint(17) + b"\x28\xb5\x2f\xfd" + b"\xff" * 12
    assert codec.decode(frame) == b"hello hello hello"
    with pytest.raises(CorruptPackage, match="ZstdError"):
        codec.decode(damaged)
