"""A mode's payload: text <-> bytes, whatever the mode is made of (the choose-once-at-the-root rule).

Why: ``compress`` and ``decompress`` used to branch on ``kind == "generic"`` - two flows that knew
which product they served. Now the selector hands every mode over as one ``PayloadCodecPort``: a
model mode codes through the stream skeleton (``stream.py``) and its drawers, a generic mode through
its codec drawer with the declared-size guard. The flows only call ``encode`` / ``decode``. Pure
plumbing (the change-without-code rule: "the skeleton"): the plates live in the drawers each side is built from.
"""

from __future__ import annotations

from hebzip.core.errors import DECODE_ERRORS, CorruptPackage
from hebzip.domain.contracts import GenericCodecPort, ModeParts
from hebzip.domain.stream import decode_payload, encode_payload
from hebzip.domain.varint import decode_varint


class ModelPayload:
    """A model mode: template + words on one range coder."""

    name = "model"

    def __init__(self, parts: ModeParts) -> None:
        self.parts = parts

    def encode(self, text: str) -> bytes:
        """Text -> payload."""
        return encode_payload(text, self.parts)

    def decode(self, payload: bytes, limit: int) -> bytes:
        """Payload -> the original UTF-8 bytes; damage -> ``CorruptPackage``."""
        try:
            return decode_payload(payload, self.parts, limit).encode("utf-8")
        except DECODE_ERRORS as error:
            raise CorruptPackage(f"הפענוח נכשל - החבילה פגומה ({type(error).__name__})") from error


class GenericPayload:
    """A generic mode: a general-purpose codec frame."""

    name = "generic"

    def __init__(self, codec: GenericCodecPort) -> None:
        self.codec = codec

    def encode(self, text: str) -> bytes:
        """Text -> codec frame of its UTF-8 bytes."""
        return self.codec.encode(text.encode("utf-8"))

    def decode(self, payload: bytes, limit: int) -> bytes:
        """Frame -> bytes; a frame that declares more than ``limit`` bytes is refused before decoding."""
        if not payload:
            raise CorruptPackage("מסגרת כללית ריקה")
        n, _pos = decode_varint(payload, 1)
        if n > limit:
            raise CorruptPackage(f"הגודל המוצהר ({n}) גדול מהמותר ({limit})")
        return self.codec.decode(payload)
