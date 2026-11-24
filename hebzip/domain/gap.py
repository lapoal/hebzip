"""Gap-classifier drawers: ``punctuation`` (mode max) and the plate ``none``.

Why a gap class at all (measured in the research prototype): the first word of a sentence, or the word right
after a comma or a quote, is predicted much better by lists of "words that start sentences" than by
the previous word. The gap = the template bytes between the previous slot and this one.
Classes (the language rows decide which characters, the mode row decides the groups):
  0             mid-sentence: nothing but ``gap_space`` (or nothing)
  1             sentence start: a ``sentence_end`` character anywhere in the gap, or an empty gap
                before the first word
  2 .. 1+M      the LAST non-space character is in ``gap_marks[i]`` (comma-like, colon, quotes, ...)
  2+M           any other last character
The order of these tests is the research prototype's (``gap_class``): sentence end first, then trailing
space stripped, then the last character. Bytes that are not UTF-8 (template codes) are skipped.
The plate answers "mid" always: predictors that react to gaps then never fire.
"""

from __future__ import annotations

from hebzip.core.errors import ConfigError
from hebzip.domain.contracts import MID_GAP, Gap

SENTENCE = 1
FIRST_MARK_CLASS = 2


class PunctuationGap:
    """Classes from the language's punctuation rows, groups from the mode's row."""

    name = "punctuation"

    def __init__(self, sentence_end: tuple[str, ...], space: str, marks: tuple[str, ...],
                 groups: tuple[int, ...]) -> None:
        if len(groups) != self.classes_for(marks):
            raise ConfigError(f"gap: {self.classes_for(marks)} classes need as many groups, got {len(groups)}")
        self.sentence_end = frozenset(sentence_end)
        self.space = space
        self.marks = marks
        self.gaps = tuple(Gap(c, g) for c, g in enumerate(groups))
        self.classes = len(groups)
        self.groups = max(groups) + 1

    @staticmethod
    def classes_for(marks: tuple[str, ...]) -> int:
        """mid + sentence + one per mark set + other."""
        return FIRST_MARK_CLASS + len(marks) + 1

    def classify(self, gap: bytes, at_start: bool) -> Gap:
        """The gap's class (see the module text) with its group."""
        s = gap.decode("utf-8", "ignore")
        if any(ch in self.sentence_end for ch in s):
            return self.gaps[SENTENCE]
        r = s.rstrip(self.space)
        if not r:
            return self.gaps[SENTENCE] if at_start else self.gaps[0]
        last = r[-1]
        for i, marks in enumerate(self.marks):
            if last in marks:
                return self.gaps[FIRST_MARK_CLASS + i]
        return self.gaps[-1]


class NoGap:
    """The plate: every gap is "mid" (class 0, group 0)."""

    name = "none"
    classes = 1
    groups = 1

    def classify(self, gap: bytes, at_start: bool) -> Gap:
        """Always mid."""
        return MID_GAP
