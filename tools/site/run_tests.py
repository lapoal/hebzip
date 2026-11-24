"""Run the website build's tests: ``python tools/site/run_tests.py`` (from any folder). Exit 0 = all passed."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "tests")],
                          cwd=HERE, check=False)
    sys.exit(done.returncode)
