"""The website build's tests import the build (tools/site) and the package (repository root) directly."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
