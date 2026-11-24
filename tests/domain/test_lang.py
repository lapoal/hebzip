"""The language pack (the change-without-code rule): a language is rows - including a language with no marks at all.

Why: ``marks = []`` - true of most alphabets - built the regex ``[]``, which does
not compile, so such a language could not be configured without changing code.
"""

from __future__ import annotations

from hebzip.domain.dictionary import WordList
from hebzip.domain.lang import LanguagePack
from hebzip.domain.tokenizer import OverlayTokenizer

LATIN = LanguagePack("plain", ((0x61, 0x7A),), (), 0xFF, 0xFE, 0xFC, 0xFD)
TEXT = "the cat sat on the mat, and the dog sat too."


def test_a_language_without_marks_tokenizes_and_round_trips() -> None:
    tokenizer = OverlayTokenizer(LATIN)
    words = WordList(["the", "cat", "sat", "on", "mat"])
    template, ids = tokenizer.split(TEXT, words)
    assert ids == [0, 1, 2, 3, 0, 4, 0, 2]
    assert tokenizer.join(template, [words.word(i) for i in ids]) == TEXT


def test_with_no_marks_nothing_is_a_mark() -> None:
    pattern = LATIN.mark_pattern()
    assert not any(pattern.match(chr(c)) for c in range(0x30000))
