"""Entry point: ``python -m atar.bench`` (see ``atar/bench/cli.py``)."""

from __future__ import annotations

import sys

from atar.bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
