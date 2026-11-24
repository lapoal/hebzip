"""Tokenizer drawers: ``overlay`` (dictionary words + a niqqud overlay) and the plate ``literal``.

Why an overlay (from the research prototype): niqqud used to split words in two, both halves went into the
template as literal text. A token is a maximal run of letters AND marks; the dictionary is asked for
the letters-only form; the marks stay in the template as a small overlay, so decoding restores the
exact original bytes. Template codes (rows, never valid UTF-8):
  slot_plain                                   dictionary word, no marks
  slot_marked g0 sep g1 sep ... marked_end     word with marks; g_i = marks after letter i (UTF-8)
Damage handling: every way a decoded template can disagree with its words is a ``CorruptPackage``.
The plate makes no slots at all: everything is literal template text (lossless, only bigger).
"""

from __future__ import annotations

from hebzip.core.errors import CorruptPackage
from hebzip.domain.contracts import DictionaryPort
from hebzip.domain.lang import LanguagePack


class OverlayTokenizer:
    """Letters-only dictionary words in slots, marks kept in the template."""

    name = "overlay"

    def __init__(self, lang: LanguagePack) -> None:
        self.lang = lang
        self.token = lang.token_pattern()
        self.mark = lang.mark_pattern()
        self.slot_end = frozenset((lang.slot_plain, lang.marked_end))

    def _overlay(self, token: str) -> tuple[str, list[str]] | None:
        """(letters-only word, marks after each letter), or None when the token starts with a mark."""
        if self.mark.match(token):
            return None
        letters: list[str] = []
        groups: list[str] = []
        for ch in token:
            if self.mark.match(ch):
                groups[-1] += ch
            else:
                letters.append(ch)
                groups.append("")
        return "".join(letters), groups

    def _encode_overlay(self, groups: list[str]) -> bytes:
        last = max(i for i, g in enumerate(groups) if g)
        sep = bytes([self.lang.mark_separator])
        body = sep.join(g.encode("utf-8") for g in groups[:last + 1])
        return bytes([self.lang.slot_marked]) + body + bytes([self.lang.marked_end])

    def split(self, text: str, d: DictionaryPort) -> tuple[bytes, list[int]]:
        """Text -> (template, word ids)."""
        out = bytearray()
        ids: list[int] = []
        pos = 0
        for m in self.token.finditer(text):
            out += text[pos:m.start()].encode("utf-8")
            pos = m.end()
            ov = self._overlay(m.group())
            wid = None if ov is None else d.lookup(ov[0])
            if ov is None or wid is None:
                out += m.group().encode("utf-8")
                continue
            ids.append(wid)
            out += self._encode_overlay(ov[1]) if any(ov[1]) else bytes([self.lang.slot_plain])
        out += text[pos:].encode("utf-8")
        return bytes(out), ids

    def _pieces(self, template: bytes) -> list[bytes | list[str]]:
        """Template -> literal byte runs and slots (a slot = its mark groups, [] when plain)."""
        pieces: list[bytes | list[str]] = []
        i, lit = 0, bytearray()
        plain, marked = self.lang.slot_plain, self.lang.slot_marked
        while i < len(template):
            b = template[i]
            if b not in (plain, marked):
                lit.append(b)
                i += 1
                continue
            if lit:
                pieces.append(bytes(lit))
                lit = bytearray()
            if b == plain:
                pieces.append([])
                i += 1
                continue
            j = template.find(bytes([self.lang.marked_end]), i)
            if j < 0:
                raise CorruptPackage("החבילה פגומה: סימון ניקוד בלי סוף")
            sep = bytes([self.lang.mark_separator])
            pieces.append([g.decode("utf-8") for g in template[i + 1:j].split(sep)])
            i = j + 1
        if lit:
            pieces.append(bytes(lit))
        return pieces

    def join(self, template: bytes, words: list[str]) -> str:
        """Fill the slots in order; any mismatch between template and words = damaged package."""
        out = bytearray()
        it = iter(words)
        for piece in self._pieces(template):
            if isinstance(piece, bytes):
                out += piece
                continue
            word = next(it, None)
            if word is None or len(piece) > len(word):
                raise CorruptPackage("החבילה פגומה: התבנית והמילים אינן מתאימות")
            groups = piece + [""] * (len(word) - len(piece))
            out += "".join(c + g for c, g in zip(word, groups, strict=True)).encode("utf-8")
        if next(it, None) is not None:
            raise CorruptPackage("החבילה פגומה: יותר מילים ממקומות בתבנית")
        return out.decode("utf-8")

    def slot_start(self, template: bytes | bytearray, after: int, end: int) -> int:
        """A plain slot is its one byte; a marked slot starts at the last ``slot_marked`` before ``end``."""
        if template[end] == self.lang.slot_plain:
            return end
        start = template.rfind(bytes([self.lang.slot_marked]), after, end)
        return start if start >= 0 else end  # only in a damaged package; ``join`` refuses it later


class LiteralTokenizer:
    """The plate: no word slots - the whole text is template (lossless, only bigger)."""

    name = "literal"
    slot_end: frozenset[int] = frozenset()

    def split(self, text: str, d: DictionaryPort) -> tuple[bytes, list[int]]:
        """Text -> (its UTF-8 bytes, no ids)."""
        return text.encode("utf-8"), []

    def join(self, template: bytes, words: list[str]) -> str:
        """The template is the text."""
        if words:
            raise CorruptPackage("החבילה פגומה: מילים בלי מקומות")
        return template.decode("utf-8")

    def slot_start(self, template: bytes | bytearray, after: int, end: int) -> int:
        """There are no slots; a slot of one byte (never asked)."""
        return end
