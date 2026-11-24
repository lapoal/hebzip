"""The language pack as the domain sees it: letters, marks and template codes - from config rows.

Why rows (design decision): a new language must need rows + tables only. Everything the tokenizer
(and, for mode max, the gap classifier and the prefix classes) knows about Hebrew is here, filled
from ``[languages.<name>]``; nothing in the code names a Hebrew letter.
The template codes must be bytes that never occur in UTF-8 (0xF8..0xFF): then a literal text byte
can never be mistaken for a slot, and no escaping is needed (checked at start-up).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Ranges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LanguagePack:
    """One language's rows."""

    name: str
    letters: Ranges
    marks: Ranges
    slot_plain: int
    slot_marked: int
    mark_separator: int
    marked_end: int
    sentence_end: tuple[str, ...] = ()
    prefixes: str = ""
    gap_space: str = ""
    gap_marks: tuple[str, ...] = ()

    def alphabet(self) -> str:
        """Every letter code point in order - the letter codes of the speller (1..len, 0 = word end)."""
        return "".join(chr(c) for a, b in self.letters for c in range(a, b + 1))

    def char_class(self, ranges: Ranges) -> str:
        """A regex character class body for code-point ranges."""
        return "".join(re.escape(chr(a)) + ("" if a == b else "-" + re.escape(chr(b))) for a, b in ranges)

    def token_pattern(self) -> re.Pattern[str]:
        """A maximal run of letters and marks = one token."""
        return re.compile(f"[{self.char_class(self.letters)}{self.char_class(self.marks)}]+")

    def mark_pattern(self) -> re.Pattern[str]:
        """One mark; a language with no marks gets a pattern that never matches.

        Why: ``[]`` is not a regex, so ``marks = []`` - true of most alphabets -
        could not be configured without changing code."""
        return re.compile(f"[{self.char_class(self.marks)}]" if self.marks else "(?!)")
