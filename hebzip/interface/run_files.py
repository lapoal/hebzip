"""File in -> file out: the one compress / decompress flow of the command line AND the loaded service.

Why one flow for both ("stay loaded" for the comparison page): the service must write the
very same package bytes as ``python -m hebzip compress`` for the same input, mode and config. Both entry
points call these two functions with an ``Assembly``; the CLI makes a fresh one per process, the service
keeps one. Nothing else differs - the same guard, the same size-limited read, the same service call
(``compress_bytes`` with its round-trip check), the same atomic write, the same numbers-only log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hebzip.config.load import load_config, project_root
from hebzip.core.clock import monotonic
from hebzip.core.errors import InputError
from hebzip.core.files import read_bytes_limited, write_bytes_atomic
from hebzip.core.logging import configure_stderr, get_logger
from hebzip.services.compress import compress_bytes
from hebzip.services.decompress import decompress_bytes
from hebzip.services.select import Assembly

LOG = get_logger("cli")
UNEXPECTED = "שגיאה לא צפויה - הפרטים ביומן (stderr)"
"""What the user reads after an unexpected failure (CLI line and service reply alike)."""


def assembly(config: Path) -> Assembly:
    """The config read and checked, the log sink on stderr, and one lazy ``Assembly``."""
    settings = load_config(config)
    configure_stderr(settings.hebzip.log_level)
    return Assembly(settings, project_root(config))


@dataclass(frozen=True)
class Done:
    """What one file run did - sizes, the mode that wrote the package, seconds. Never content."""

    before: int
    after: int
    mode: str
    seconds: float


def guard_output(output: Path, force: bool) -> None:
    """An existing output is kept unless ``force`` (exit 7)."""
    if output.exists() and not force:
        raise InputError(f"קובץ הפלט כבר קיים: {output} (השתמש ב---force כדי לדרוס)")


def compress_file(asm: Assembly, source: Path, output: Path, mode: str | None, use_fallback: bool,
                  force: bool) -> Done:
    """``source`` (UTF-8 text) -> package file ``output``; ``mode`` None = ``default_mode``."""
    guard_output(output, force)
    started = monotonic()
    raw = read_bytes_limited(source, asm.settings.hebzip.max_input_bytes)
    result = compress_bytes(asm, raw, mode or asm.settings.hebzip.default_mode, use_fallback)
    write_bytes_atomic(output, result.data)
    done = Done(len(raw), len(result.data), result.mode, round(monotonic() - started, 2))
    LOG.info("done", mode=done.mode, before=done.before, after=done.after, seconds=done.seconds)
    return done


def decompress_file(asm: Assembly, source: Path, output: Path, force: bool) -> Done:
    """Package file ``source`` -> the original bytes in ``output``."""
    guard_output(output, force)
    started = monotonic()
    data = read_bytes_limited(source, asm.settings.hebzip.max_output_bytes)
    raw = decompress_bytes(asm, data)
    write_bytes_atomic(output, raw)
    done = Done(len(data), len(raw), "", round(monotonic() - started, 2))
    LOG.info("done", before=done.before, after=done.after, seconds=done.seconds)
    return done
