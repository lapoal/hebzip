"""Typed errors - the only exceptions hebzip raises on purpose.

Why typed (the no-broad-except rule): no bare ``except Exception``; each layer catches the class it
understands and the CLI turns it into a message plus an exit code. Messages are plain Hebrew for
the user and never carry the user's text (the no-user-content-in-metrics rule) - only sizes, names, short hashes.
"""

from __future__ import annotations


class HebzipError(Exception):
    """Base of every deliberate failure. ``str(error)`` is Hebrew; ``exit_code`` is the CLI code."""

    exit_code = 1


class ConfigError(HebzipError):
    """A row of hebzip.toml is missing, unknown, invalid, or breaks a dependency."""

    exit_code = 2


class TableMismatch(HebzipError):
    """A table's sha256 differs from the config, or a package was made with other tables."""

    exit_code = 3


class TablesMissing(TableMismatch):
    """The pinned table files a mode needs are not on this disk (a fresh clone: the tables are release
    downloads, not part of the repository) - the message says where to get them (``store/missing.py``)."""


class NarrowingOverflow(TableMismatch):
    """A table value fits none of the allowed integer widths (``[hebzip].narrow_bits``) - refused when the
    derived copy is made, never cut to fit."""


class CorruptPackage(HebzipError):
    """Bad magic / varint / unknown mode / decoding failed / declared size too large."""

    exit_code = 4


class ChecksumMismatch(CorruptPackage):
    """The package carries a checksum and the payload does not match it (integrity on)."""


class UnsupportedFormat(HebzipError):
    """The header version is not one this code can read."""

    exit_code = 4


class DamagedStream(CorruptPackage):
    """The decoder read past the end of the body beyond ``phantom_max``, or its interval broke.

    Design decision: a damaged package must end in bounded work with a typed error."""


class RoundTripFailed(HebzipError):
    """The in-process verification failed - a bug; no output is written."""

    exit_code = 5


class DrawerUnavailable(HebzipError):
    """A null plate was asked to work (the safe direction: refuse, never guess)."""

    exit_code = 6


class EngineOrderError(HebzipError):
    """The kernels of this process were already built in another engine mode (a wiring bug)."""

    exit_code = 1


class InputError(HebzipError):
    """Missing / too big / not UTF-8 input, or the output exists without --force."""

    exit_code = 7


BUG_KINDS: tuple[type[Exception], ...] = (
    LookupError, TypeError, ValueError, AttributeError, ArithmeticError, OSError, RuntimeError,
    AssertionError, ImportError, NameError)
"""What the CLI root safety net catches instead of a blind ``except Exception`` (the root safety-net rule)."""

DECODE_ERRORS: tuple[type[Exception], ...] = (UnicodeDecodeError,)
"""What a damaged package can still raise from the standard library after decoding (the bytes are
not UTF-8); everything else the decoder raises is already a ``CorruptPackage``."""
