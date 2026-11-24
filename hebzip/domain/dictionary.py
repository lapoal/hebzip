"""Dictionary drawers: ``word_list`` (the language pack's word list) and the plate ``null``.

Why the id of a word is its line number: the base prior and the corpus tables are indexed by the
same ids, so the word list is part of the format (pinned by sha256 like every table). The word ->
id index is built only when encoding (the decoder never needs it); a word that appears twice keeps
its LAST line, as the research prototype's ``dict(zip(words, range))`` did.
The plate knows no word: the tokenizer then leaves every token literal (lossless, only bigger), and
asking it for a word id refuses (``DrawerUnavailable``) instead of guessing.
"""

from __future__ import annotations

from hebzip.core.errors import CorruptPackage, DrawerUnavailable


class WordList:
    """Word ids = positions in the language pack's word list."""

    name = "word_list"

    def __init__(self, words: list[str]) -> None:
        self.words = words
        self._index: dict[str, int] | None = None

    def size(self) -> int:
        """Number of words."""
        return len(self.words)

    def lookup(self, word: str) -> int | None:
        """The id of ``word`` (index built on first use)."""
        if self._index is None:
            self._index = dict(zip(self.words, range(len(self.words)), strict=True))
        return self._index.get(word)

    def word(self, wid: int) -> str:
        """The word with id ``wid``; an id outside the list = damaged package."""
        if not 0 <= wid < len(self.words):
            raise CorruptPackage(f"החבילה פגומה: מספר מילה {wid} מחוץ למילון")
        return self.words[wid]


class NullDictionary:
    """The plate: knows no word."""

    name = "null"

    def size(self) -> int:
        """No words."""
        return 0

    def lookup(self, word: str) -> int | None:
        """Never found - the token stays literal."""
        return None

    def word(self, wid: int) -> str:
        """Refuses - guessing a word would decode wrongly."""
        raise DrawerUnavailable("dictionary")
