"""``python -m hebzip`` - the tested way in."""

from __future__ import annotations

import sys

from hebzip.interface.cli import main

if __name__ == "__main__":
    sys.exit(main())
