"""Original bytes -> hebzip package: chosen mode, fallback, round-trip proof, header.

The text is compressed as ``str`` and the original bytes must come back exactly (``\\r\\n`` and a
BOM included) - ``verify_roundtrip`` proves it in-process before anything is written; a failure is
a bug (``RoundTripFailed``) and no output exists. Fallback (config ``fallback`` + ``fallback_policy``):
"smaller" encodes the fallbacks too and keeps the smallest (tie = the primary); "on_error" only
falls back when the primary refused with ``DrawerUnavailable`` (a null plate was asked to work).
"""

from __future__ import annotations

from dataclasses import dataclass

from hebzip.core.errors import ConfigError, DrawerUnavailable, InputError, RoundTripFailed
from hebzip.core.logging import get_logger
from hebzip.domain.header import build_header
from hebzip.services.decompress import decompress_bytes
from hebzip.services.select import Assembly

LOG = get_logger("compress")


@dataclass(frozen=True)
class Compressed:
    """The finished package and which mode produced it."""

    data: bytes
    mode: str
    raw_size: int


def _package(asm: Assembly, mode: str, text: str) -> bytes:
    """One complete package (header + payload) of ``mode``."""
    spec = asm.spec(mode)
    sig = asm.signature(mode)
    payload = asm.payload(mode).encode(text)
    integrity = asm.integrity(asm.settings.hebzip.integrity)
    return build_header(spec.mode_id, sig, integrity.code, integrity.compute(payload)) + payload


def _try(asm: Assembly, mode: str, text: str) -> bytes | None:
    try:
        return _package(asm, mode, text)
    except DrawerUnavailable as error:
        LOG.warning("drawer_unavailable", mode=mode, drawer=str(error))
        return None


def _smallest(asm: Assembly, names: list[str], text: str,
              best: tuple[bytes, str] | None = None) -> tuple[bytes, str] | None:
    """The smallest package among ``best`` and ``names`` (tie = the earlier); ``None`` when all refused."""
    for name in names:
        candidate = _try(asm, name, text)
        if candidate is not None and (best is None or len(candidate) < len(best[0])):
            best = (candidate, name)
    return best


def _choose(asm: Assembly, mode: str, text: str, fallbacks: list[str]) -> tuple[bytes, str]:
    """The primary = the smallest of the mode's candidates (itself, or a choose mode's list); then
    the fallbacks per ``fallback_policy``."""
    best = _smallest(asm, asm.candidates(mode), text)
    if best is not None and not asm.fallback_always():
        return best
    best = _smallest(asm, fallbacks, text, best)
    if best is None:
        raise DrawerUnavailable(f"{mode}: no drawer could produce a package (fallbacks tried: {len(fallbacks)})")
    return best


def compress_bytes(asm: Assembly, raw: bytes, mode: str, use_fallback: bool = True) -> Compressed:
    """Compress ``raw`` (UTF-8 text) with ``mode``; fall back per the config unless disabled."""
    if asm.spec(mode).retired:
        raise ConfigError(f"[modes.{mode}]: retired - decoding only")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InputError("הקלט אינו UTF-8 תקין") from error
    fallbacks = [n for n in asm.settings.hebzip.fallback if n != mode] if use_fallback else []
    data, used = _choose(asm, mode, text, fallbacks)
    if asm.settings.hebzip.verify_roundtrip and decompress_bytes(asm, data) != raw:
        raise RoundTripFailed(f"{used}: הפענוח לא החזיר את הקלט המקורי ({len(raw)} בתים)")
    LOG.info("compressed", mode=used, before=len(raw), after=len(data))
    return Compressed(data, used, len(raw))
