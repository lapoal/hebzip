"""The template kernel's precision is two config rows (a design decision: "zero exceptions").

``[modes.<m>.template].prob_bits`` (template-bit probabilities, 12 = 1..4095 of 4096) and ``fix_bits`` (mixer
weights / adaptive probabilities in fixed point, 16 = 16.16) used to be constants inside the kernel. Through
the real CLI in child processes (the real-path rule):
(b) another precision still round-trips exactly, and the package differs from the default one;
(c) a package made with the default precision is refused (exit 3) by a config with another precision -
    never decoded to garbage;
(d) an illegal precision refuses to start (exit 2), naming the row, and writes nothing.
(a) - default rows = golden bytes - is ``tests/golden/test_golden_identity.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hebzip.core.errors import ConfigError, TableMismatch
from tests.support import compress_file, config_variant, decompress_file, sample_paths, work_dir

FAST_PROB = "prob_bits = 12                   # template-bit probabilities: 12-bit integers (1..4095 of 4096)"
FAST_FIX = "fix_bits = 16                    # mixer weights 16.16, adaptive probabilities 16-bit"
CHANGED = {"prob_13": (FAST_PROB, FAST_PROB.replace("= 12", "= 13")),
           "fix_17": (FAST_FIX, FAST_FIX.replace("= 16", "= 17"))}
ILLEGAL = {"prob_0": ((FAST_PROB, FAST_PROB.replace("= 12", "= 0")), "prob_bits"),
           "fix_below_prob": ((FAST_FIX, FAST_FIX.replace("= 16", "= 11")), "fix_bits"),
           "prob_too_small_for_knots": ((FAST_PROB, FAST_PROB.replace("= 12", "= 11")), "squash_knots")}


def _compress(text: Path, out: Path, *extra: str) -> bytes:
    done = compress_file(text, out, "fast", "--no-fallback", *extra)
    assert done.returncode == 0, done.stderr[-400:]
    return out.read_bytes()


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_another_precision_round_trips_exactly_and_changes_the_package(name: str) -> None:
    work = work_dir(f"precision_{name}")
    cfg = config_variant(f"precision_{name}_cfg", CHANGED[name])
    for text in sample_paths():
        default = _compress(text, work / f"{text.stem}.default.hz")
        pkg = work / f"{text.stem}.hz"
        changed = _compress(text, pkg, "--config", str(cfg))
        assert changed[2:] != default[2:], f"{text.stem}: the precision row changed nothing"
        out = work / f"{text.stem}.out"
        done = decompress_file(pkg, out, "--config", str(cfg))
        assert done.returncode == 0, done.stderr[-400:]
        assert out.read_bytes() == text.read_bytes(), f"{text.stem}: round trip differs"


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_a_default_package_is_refused_under_another_precision(name: str) -> None:
    work = work_dir(f"precision_refuse_{name}")
    cfg = config_variant(f"precision_refuse_{name}_cfg", CHANGED[name])
    text = sample_paths()[0]
    pkg, out = work / "x.hz", work / "x.out"
    _compress(text, pkg)
    done = decompress_file(pkg, out, "--config", str(cfg))
    assert done.returncode == TableMismatch.exit_code, done.stderr[-400:]
    assert not out.exists()


@pytest.mark.parametrize("name", sorted(ILLEGAL))
def test_an_illegal_precision_refuses_to_start(name: str) -> None:
    replacement, row = ILLEGAL[name]
    work = work_dir(f"precision_illegal_{name}")
    cfg = config_variant(f"precision_illegal_{name}_cfg", replacement)
    out = work / "x.hz"
    done = compress_file(sample_paths()[0], out, "fast", "--config", str(cfg))
    assert done.returncode == ConfigError.exit_code, done.stderr[-400:]
    assert f"[modes.fast.template].{row}" in done.stderr.decode("utf-8")
    assert not out.exists()
