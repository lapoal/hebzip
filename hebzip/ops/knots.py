"""The ``squash_knots`` row of the template model, computed from the logistic function.

    python -m hebzip.ops.knots [--prob-bits 12] [--count 33]

Why a generator: the knot values are
plain samples of a published mathematical function, made here, so anyone can check where every
number in the row comes from. The row itself stays in the config (encoder and decoder read the
row, never this file); ``tests/ops/test_knots.py`` checks that every ``squash_knots`` row in the
config equals what this tool prints.

The definition. With ``one = 2**prob_bits`` and ``count = 2**m + 1`` knots, knot i (0 .. count-1)
sits at the stretch value d_i = i * one / (count - 1) - one / 2, so the knots cover -one/2 .. +one/2.
The stretch domain is read as natural log-odds scaled so that its two ends are -/+ LOGIT_RANGE:

    value_i = one / (1 + exp(-LOGIT_RANGE * 2 * d_i / one))

Rounding: to the nearest integer, an exact half upwards (``floor(value + 1/2)``), then held inside
the valid probabilities 1 .. one-1. The curve is symmetric, so knot i and knot count-1-i add up
to ``one`` (the middle knot is exactly one/2). This is a tool: it writes the row to stdout directly.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

LOGIT_RANGE = 8
"""The log-odds at the two ends of the stretch domain: +-8 is a probability of 1 - 1/(1 + e**8),
about 0.99966 - beyond that a 12-bit probability cannot move anyway."""


def logistic_knots(prob_bits: int, count: int) -> list[int]:
    """The ``count`` knot values for probabilities of ``prob_bits`` bits (see the module text)."""
    one = 1 << prob_bits
    spans = count - 1
    if spans < 2 or spans & (spans - 1) or spans > one:
        raise ValueError("count must be 2**m + 1 knots, m = 1 .. prob_bits")
    knots = []
    for i in range(count):
        d = i * one / spans - one / 2
        value = one / (1.0 + math.exp(-LOGIT_RANGE * 2 * d / one))
        knots.append(min(max(math.floor(value + 0.5), 1), one - 1))
    return knots


def toml_row(knots: list[int]) -> str:
    """The row as it is written in ``config/hebzip.toml`` (19 values on the first line)."""
    first, rest = knots[:19], knots[19:]
    head = "squash_knots = [" + ", ".join(map(str, first)) + ","
    return head + "\n                " + ", ".join(map(str, rest)) + "]"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the row for the given precision."""
    parser = argparse.ArgumentParser(prog="hebzip.ops.knots")
    parser.add_argument("--prob-bits", type=int, default=12)
    parser.add_argument("--count", type=int, default=33)
    args = parser.parse_args(argv)
    sys.stdout.write(toml_row(logistic_knots(args.prob_bits, args.count)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
