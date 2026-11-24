"""The payload skeleton (plain code, the change-without-code rule): template bytes and word slots on one range coder.

payload = varint(template length) + coded body. Template bytes are coded one by one; right after a
byte that ends a slot the slot's word is coded on the same coder, and the template model learns
which word came (``set_word``). This is the research prototype's loop (``tp_package_fast._code``) without
its first byte: the mode lives in the header now.
Before every word the gap in front of it (the template bytes since the previous slot) goes through
the gap classifier drawer (the only change here for mode max; fast's plate says "mid").
Damage: the declared length is capped by the caller; every kernel step checks the decoder budget,
so a damaged package ends in bounded work with ``DamagedStream`` (design decision).
"""

from __future__ import annotations

from hebzip.core.errors import CorruptPackage
from hebzip.domain.coder import RangeDecoder, RangeEncoder
from hebzip.domain.contracts import Gap, GapClassifierPort, ModeParts, TokenizerPort
from hebzip.domain.varint import decode_varint, encode_varint


def gap_before(tokenizer: TokenizerPort, gap: GapClassifierPort, buf: bytes | bytearray, after: int, end: int,
               first: bool) -> Gap:
    """The class of the bytes between the previous slot (ending before ``after``) and the slot ending at ``end``.

    Public because the language-pack builder (``ops/counts.py``) must classify a corpus exactly
    as the coder does - one function for both, never a copy."""
    return gap.classify(bytes(buf[after:tokenizer.slot_start(buf, after, end)]), first)


def _gap(p: ModeParts, buf: bytes | bytearray, after: int, end: int, first: bool) -> Gap:
    return gap_before(p.tokenizer, p.gap, buf, after, end, first)


def encode_payload(text: str, p: ModeParts) -> bytes:
    """Text -> payload."""
    template, ids = p.tokenizer.split(text, p.dictionary)
    coder = RangeEncoder(p.kernels)
    p.template.start(len(template))
    p.words.start()
    j = after = 0
    for i, byte in enumerate(template):
        p.template.code_byte(coder, byte)
        if byte in p.tokenizer.slot_end:
            gap = _gap(p, template, after, i, j == 0)
            after = i + 1
            wid = p.words.code(coder, ids[j], gap)
            p.template.set_word(wid)
            j += 1
    return encode_varint(len(template)) + coder.finish()


def decode_payload(payload: bytes, p: ModeParts, max_template: int) -> str:
    """Payload -> text; a declared template longer than ``max_template`` is refused at once."""
    n, pos = decode_varint(payload, 0)
    if n > max_template:
        raise CorruptPackage(f"אורך התבנית המוצהר ({n}) גדול מהמותר ({max_template})")
    coder = RangeDecoder(p.kernels, payload[pos:], p.phantom_max)
    coder.check()
    p.template.start(n)
    p.words.start()
    out = bytearray()
    wids: list[int] = []
    after = 0
    for i in range(n):
        byte = p.template.code_byte(coder, 0)
        out.append(byte)
        if byte in p.tokenizer.slot_end:
            gap = _gap(p, out, after, i, not wids)
            after = i + 1
            wid = p.words.code(coder, 0, gap)
            p.template.set_word(wid)
            wids.append(wid)
    return p.tokenizer.join(bytes(out), [p.dictionary.word(w) for w in wids])
