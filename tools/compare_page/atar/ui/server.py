"""Start the comparison page on 127.0.0.1 and open the browser - the one double-click start.

Stdlib wsgiref,
threaded so polling never waits behind an upload; already running on the port -> just open
the browser. Added here: leftovers in the material folder are wiped before serving
(the machine-and-material rule), and a config error stops the start with its Hebrew sentence.
Closing the console window ends this process, and with it the server.
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from atar.compressors.loaded import ServicePool
from atar.compressors.registry import build
from atar.config.load import DEFAULT_CONFIG, Settings, load
from atar.core.errors import ConfigError
from atar.core.logging import configure_file_logging, get_logger
from atar.services import material
from atar.services.jobs import JobBoard
from atar.ui import lifecycle
from atar.ui.app import APP_NAME, build_app

_log = get_logger("server")


class ThreadingServer(socketserver.ThreadingMixIn, WSGIServer):
    """One thread per request; threads die with the server."""

    daemon_threads = True


class QuietHandler(WSGIRequestHandler):
    """No per-request console lines (a query string carries the uploaded file's name)."""

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def say(lines: Sequence[str]) -> None:
    """The console output of this entry point (the only place that writes to it).

    UTF-8 when piped: the Windows default code page here (cp1255) cannot write every
    character, and a crash while printing a Hebrew message would hide the message itself.
    """
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def already_ours(url: str) -> bool:
    """True when our page already answers on this address."""
    try:
        with urllib.request.urlopen(url + "/api/ping", timeout=2) as reply:
            return bool(json.loads(reply.read().decode("utf-8")).get("app") == APP_NAME)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def make_board(settings: Settings) -> JobBoard:
    """Every configured row with its drawer, the services those rows use (not started yet), the material."""
    pool = ServicePool(settings.services)
    tools = [(row, build(row, pool)) for row in settings.tools]
    return JobBoard(tools, settings.material, settings.max_text_bytes, pool)


def serve(settings: Settings, open_browser: bool) -> int:
    """Wipe leftovers, bind, start the loaded tools in the background, serve until the window closes.

    The services start only AFTER the port is ours (a second start.bat must not load a second copy),
    and in the background (the page opens at once and shows "טוען את הכלי…"). When this process ends
    - window closed, Ctrl+C, crash - each service sees its stdin close and ends by itself; on a normal
    stop ``stop_all`` also waits for that and kills what is left.
    """
    url = f"http://{settings.host}:{settings.port}"
    if already_ours(url):
        say([f"הדף כבר פועל: {url}"])
        if open_browser:
            webbrowser.open(url)
        return 0
    configure_file_logging(settings.logs / "atar.log")
    material.wipe_all(settings.material)
    board = make_board(settings)
    app = build_app(settings, board)
    try:
        server = make_server(settings.host, settings.port, app,
                             server_class=ThreadingServer, handler_class=QuietHandler)
    except OSError:
        say([f"הפורט {settings.port} תפוס על-ידי תוכנה אחרת. שנה את port בקובץ config\\atar.toml"])
        return 2
    board.pool.start_all()
    say([f"דף ההשוואה פועל: {url}", "כדי לעצור - סגור את החלון הזה."])
    lifecycle.started(settings.logs, os.getpid(), settings.port, len(board.pool.used))
    lifecycle.on_console_close(settings.logs)
    if open_browser:
        webbrowser.open(url)
    reason = "serve_loop_ended"
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        reason = "ctrl_c"
    except BaseException as error:
        reason = f"unexpected_{type(error).__name__}"  # last-resort: loud, then re-raised
        raise
    finally:
        lifecycle.stopped(settings.logs, reason)
        server.server_close()
        board.pool.stop_all()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m atar.ui.server [--config X] [--no-browser]``."""
    p = argparse.ArgumentParser(prog="hebzip-atar")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    try:
        settings = load(args.config)
        return serve(settings, settings.open_browser and not args.no_browser)
    except ConfigError as error:
        say([str(error)])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
