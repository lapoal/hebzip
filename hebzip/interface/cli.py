"""The command line: thin - it parses, calls a service, maps errors to exit codes.

    python -m hebzip compress   --mode fast|max|generic [--no-fallback] [--config PATH] [--force] IN OUT
    python -m hebzip decompress                          [--config PATH] [--force] IN OUT
    python -m hebzip info       IN [--config PATH]
    python -m hebzip serve      [--modes a,b] [--config PATH]   (stays loaded: JSON lines, see serve.py)

Why the root safety net (the root safety-net rule): any unexpected failure ends with a friendly Hebrew
line, a LOUD log record (type + traceback to stderr) and exit code 1 - never a silent fault, never
a partial output (the write is atomic and happens only after everything succeeded). User-facing
output goes through ``sys.stdout.write`` here and nowhere else (no ``print``); the log never holds
the text, only sizes and names.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from hebzip.config.load import DEFAULT_CONFIG, load_config, project_root
from hebzip.core.errors import BUG_KINDS, HebzipError
from hebzip.core.files import read_bytes_limited
from hebzip.core.logging import configure_stderr
from hebzip.domain.header import kind_of, parse_header
from hebzip.interface import serve
from hebzip.interface.run_files import LOG, UNEXPECTED, assembly, compress_file, decompress_file, guard_output
from hebzip.services.select import Assembly


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hebzip", description="Hebrew text compression (fast / max / generic)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compress", "decompress"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--force", action="store_true", help="overwrite an existing output file")
        if name == "compress":
            p.add_argument("--mode", default=None, help="a [modes.*] name (default: default_mode)")
            p.add_argument("--no-fallback", action="store_true", help="never fall back to the generic codec")
    info = sub.add_parser("info")
    info.add_argument("input", type=Path)
    info.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    srv = sub.add_parser("serve", help="stay loaded and answer JSON-line requests on stdin (serve.py)")
    srv.add_argument("--modes", default=None, help="modes to build before 'ready', comma separated")
    srv.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _run_compress(args: argparse.Namespace) -> int:
    guard_output(args.output, args.force)  # before the config is read, as always (exit 7 first)
    compress_file(assembly(args.config), args.input, args.output, args.mode, not args.no_fallback, args.force)
    return 0


def _run_decompress(args: argparse.Namespace) -> int:
    guard_output(args.output, args.force)
    decompress_file(assembly(args.config), args.input, args.output, args.force)
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes else None
    return serve.run(args.config, modes, sys.stdin.buffer, sys.stdout)


def _run_info(args: argparse.Namespace) -> int:
    settings = load_config(args.config)
    asm = Assembly(settings, project_root(args.config))  # lazy: builds nothing, only the id lookups
    data = read_bytes_limited(args.input, settings.hebzip.max_output_bytes)
    kind = kind_of(data)
    lines = [f"size: {len(data)}", f"format: {kind}"]
    if kind == "legacy":
        lines += [f"first_byte: {data[0]:#04x}", "supported: no (raw experiment package)"]
    elif kind == "v1":
        header = parse_header(data)
        mode = asm.mode_by_id(header.mode_id) or "unknown"
        lines += [f"version: {header.version}", f"mode: {mode}", f"mode_id: {header.mode_id}",
                  f"table_sig: {header.table_sig.hex()}", f"integrity: {header.integrity}",
                  f"payload_bytes: {len(data) - header.payload_at}"]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _say(text: str) -> None:
    sys.stderr.write(text + "\n")


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit code (0 ok codes, 1 = unexpected)."""
    _utf8_streams()
    args = _parser().parse_args(argv)
    runners = {"compress": _run_compress, "decompress": _run_decompress, "info": _run_info, "serve": _run_serve}
    try:
        return runners[args.command](args)
    except HebzipError as error:
        _say(str(error))
        return error.exit_code
    except BUG_KINDS:
        configure_stderr("INFO")
        LOG.error("unexpected failure")
        _say(UNEXPECTED)
        return 1
