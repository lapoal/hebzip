"""Cold start of the real entry point in a clean interpreter (the cold-start rule).

Also proves the start-up wipe through the real path: a leftover job folder in the material
folder is gone once the server answers /api/ping.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

from atar.config.load import DEFAULT_CONFIG, ROOT
from tests.support import WORK, free_port


def test_import_in_clean_interpreter() -> None:
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, r'{ROOT}'); "
                           "import atar.ui.server"], capture_output=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")


def test_bad_config_stops_with_hebrew_message() -> None:
    bad = WORK / "bad.toml"
    bad.write_text("[server]\nhost = '0.0.0.0'\n", encoding="utf-8")
    done = subprocess.run([sys.executable, "-m", "atar.ui.server", "--config", str(bad),
                           "--no-browser"], cwd=ROOT, capture_output=True, timeout=60, check=False)
    assert done.returncode == 2
    assert "בהגדרות" in done.stdout.decode("utf-8", "replace")


def test_server_starts_and_wipes_leftovers() -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    port = free_port()
    text = text.replace("port = 8837", f"port = {port}")
    text = text.replace('material = "material"', f"material = '{WORK / 'material'}'")
    text = text.replace('logs = "logs"', f"logs = '{WORK / 'logs'}'")
    cfg = WORK / "atar.toml"
    cfg.write_text(text, encoding="utf-8")
    leftover = WORK / "material" / "job-7"
    leftover.mkdir(parents=True)
    (leftover / "text.txt").write_text("שארית", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "atar.ui.server", "--config", str(cfg),
                             "--no-browser"], cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        answer = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2) as r:
                    answer = json.loads(r.read())
                break
            except OSError:
                time.sleep(0.2)
        assert answer == {"app": "hebzip-atar"}
        assert not leftover.exists()
    finally:
        # The venv python.exe is a launcher that starts the real interpreter as a child;
        # kill() alone leaves that child serving. Kill the whole tree.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       check=False)
        proc.wait(10)
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1).close()
                time.sleep(0.2)
            except OSError:
                break
        else:
            raise AssertionError("test server still answering after kill")
