"""The root safety net's log line: loud but content-free.

An unexpected failure must be locatable from the log alone - the exception type AND where it
happened (file:line function for each frame) - while the exception MESSAGE, which may quote the
user's text (a ``KeyError`` on a word, a ``UnicodeDecodeError`` snippet), is never written.
"""

from __future__ import annotations

import json
import logging
import sys

from hebzip.core.logging import JsonFormatter

USER_TEXT = "שלום-סודי-של-המשתמש"


def _boom() -> None:
    raise KeyError(USER_TEXT)


def test_unexpected_failure_is_logged_with_frames_but_without_its_message() -> None:
    try:
        _boom()
    except KeyError:
        record = logging.LogRecord("hebzip.cli", logging.ERROR, __file__, 1, "unexpected failure", None,
                                   sys.exc_info())
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["error_type"] == "KeyError"
    assert any("test_logging.py" in frame and "_boom" in frame for frame in payload["where"]), payload
    assert USER_TEXT not in line
