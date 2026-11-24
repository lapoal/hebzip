"""Plates of the drawers OUTSIDE the heart's model: fallback codec, holes, checksum.

Why "safe": a drawer that was forgotten (or switched off) must never produce a wrong word or a
silently different package. The fallback plate refuses (``DrawerUnavailable``); holes answer "none";
the checksum plate is "no checksum" (a design decision). The heart's own plates live next
to their real drawers (``tokenizer.py``, ``dictionary.py``, ``template_cm.py``, ``words/``).
"""

from __future__ import annotations

from hebzip.core.errors import CorruptPackage, DrawerUnavailable


class NullGenericCodec:
    """No fallback codec."""

    name = "null"

    def encode(self, raw: bytes) -> bytes:
        """Refuses."""
        raise DrawerUnavailable("fallback")

    def decode(self, payload: bytes) -> bytes:
        """Refuses."""
        raise DrawerUnavailable("fallback")

    def codec_names(self) -> list[str]:
        """No codecs."""
        return []


class NullHoles:
    """No holes: ``choose`` always answers "none" - the only built value of ``holes``."""

    name = "null"

    def choose(self, slots: list[int], context: str) -> dict[int, str]:
        """No slot becomes a hole."""
        return {}


class NullIntegrity:
    """No checksum (the default): zero bytes, nothing to compare."""

    name = "none"
    code = 0
    size = 0

    def compute(self, payload: bytes) -> bytes:
        """No checksum bytes."""
        return b""


class ChoosePayload:
    """A ``choose`` mode as a payload codec: it has no payload of its own - compress packs each of its
    candidates and keeps the smallest, whose id goes in the header. A package that names a choose mode's id
    was not made by hebzip: decoding refuses (``CorruptPackage``) and encoding refuses (``DrawerUnavailable``)."""

    name = "choose"

    def encode(self, text: str) -> bytes:
        """Refuses - the flow packs the candidates instead."""
        raise DrawerUnavailable("choose")

    def decode(self, payload: bytes, limit: int) -> bytes:
        """Refuses - no package carries a choose mode."""
        raise CorruptPackage("מצב בחירה אינו מופיע בחבילה")
