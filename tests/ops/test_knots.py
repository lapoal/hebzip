"""The ``squash_knots`` rows are the logistic function, nothing else.

* every ``squash_knots`` row in the real config equals ``hebzip.ops.knots.logistic_knots`` for its mode's
  ``prob_bits`` - so the row can be traced to the published formula, and a hand-edited knot fails here;
* the knots are symmetric (knot i + knot count-1-i = 2**prob_bits) and rising, as the definition promises;
* the tool prints exactly the row that is in the config (the real entry point, in a child process).
"""

from __future__ import annotations

import subprocess
import sys

from hebzip.ops.knots import logistic_knots, toml_row
from tests.support import CONFIG, ROOT, real_config


def test_every_knot_row_is_the_logistic_function() -> None:
    modes = [m for m in real_config()["modes"].values() if "template" in m]
    assert len(modes) >= 7
    for mode in modes:
        t = mode["template"]
        assert t["squash_knots"] == logistic_knots(t["prob_bits"], len(t["squash_knots"])), mode["id"]


def test_the_knots_are_symmetric_and_rising() -> None:
    for bits, count in ((12, 33), (10, 17), (14, 65)):
        k = logistic_knots(bits, count)
        assert all(k[i] + k[count - 1 - i] == 1 << bits for i in range(count))
        assert k == sorted(k) and 1 <= k[0] and k[-1] < 1 << bits and k[count // 2] == 1 << (bits - 1)


def test_the_tool_prints_the_row_of_the_config() -> None:
    done = subprocess.run([sys.executable, "-m", "hebzip.ops.knots"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    row = done.stdout.strip()
    assert row == toml_row(logistic_knots(12, 33)) and CONFIG.read_text(encoding="utf-8").count(row) >= 7
