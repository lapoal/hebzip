"""The suite runner never crashes while reporting a failure.

Why: ``tests/run_tests.py`` printed a failing file's pytest output to a stdout whose encoding was the
Windows code page (cp1255, output redirected to a log). The output holds Hebrew error text and the
U+FFFD marks of the runner's own ``errors="replace"`` decoding; cp1255 cannot encode U+FFFD, the
runner died with ``UnicodeEncodeError`` - and the failure details of the OTHER failing files were
never printed. A failure report must never crash the runner.

The real ``main`` runs in a child process with ``PYTHONIOENCODING=cp1255`` and a piped stdout (the
same situation as the log); only the per-file pytest child is replaced by a canned failure whose text
cannot be encoded in cp1255, so the report path itself is what is tested.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tests.support import ROOT

FAILURE = "� הטבלה אינה בעותק → x"
CHILD = ("import sys; import tests.run_tests as r; "
         f"r.run_file = lambda path: (False, {FAILURE!r}); "
         "sys.exit(r.main(['test_header']))")


def test_a_failure_report_with_unencodable_text_does_not_crash_the_runner() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1255"}
    done = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, capture_output=True, env=env, timeout=120,
                          check=False)
    out, err = done.stdout.decode("cp1255", errors="replace"), done.stderr.decode("cp1255", errors="replace")
    assert "Traceback" not in err, err[-800:]
    assert done.returncode == 1, (done.returncode, err[-800:])
    assert "FAIL  domain\\test_header.py" in out or "FAIL  domain/test_header.py" in out, out
    assert "הטבלה אינה בעותק" in out and "0/1 files passed" in out, out
