"""Conformance of the dictionary and tokenizer drawers: real and plate answer the same contract, the
plate fails in the safe direction, and ``join(split(t)) == t`` on all Hebrew golden texts for both."""

from __future__ import annotations

import pytest

from hebzip.config.load import load_config
from hebzip.core.errors import CorruptPackage, DrawerUnavailable
from hebzip.domain.contracts import DictionaryPort, TokenizerPort
from hebzip.domain.dictionary import NullDictionary, WordList
from hebzip.domain.tokenizer import LiteralTokenizer, OverlayTokenizer
from hebzip.services.assemble import language
from hebzip.store.files import read_words
from tests.support import CONFIG, ROOT, assert_same_signatures, text_paths

SETTINGS = load_config(CONFIG)
LANG = language(SETTINGS.languages["he"], "he")


@pytest.fixture(scope="module")
def words() -> WordList:
    return WordList(read_words(ROOT / SETTINGS.tables["he_words"].path))


def test_dictionary_real_and_plate_share_the_contract(words: WordList) -> None:
    plate = NullDictionary()
    assert isinstance(words, DictionaryPort) and isinstance(plate, DictionaryPort)
    assert_same_signatures(DictionaryPort, words, plate)
    assert words.size() == 1175672 and words.word(words.lookup("שלום") or 0) == "שלום"
    with pytest.raises(CorruptPackage):
        words.word(words.size())
    assert plate.lookup("שלום") is None and plate.size() == 0
    with pytest.raises(DrawerUnavailable):
        plate.word(0)


def test_tokenizer_real_and_plate_share_the_contract() -> None:
    real, plate = OverlayTokenizer(LANG), LiteralTokenizer()
    assert isinstance(real, TokenizerPort) and isinstance(plate, TokenizerPort)
    assert_same_signatures(TokenizerPort, real, plate)
    assert real.slot_end == frozenset((0xFF, 0xFD)) and plate.slot_end == frozenset()


def test_join_split_is_exact_on_every_golden_text(words: WordList) -> None:
    real, plate = OverlayTokenizer(LANG), LiteralTokenizer()
    for path in text_paths():
        text = path.read_bytes().decode("utf-8")
        template, ids = real.split(text, words)
        assert ids and real.join(template, [words.word(i) for i in ids]) == text, path.stem
        assert plate.split(text, words) == (text.encode("utf-8"), [])
        assert plate.join(*plate.split(text, words)[:1], []) == text
        assert real.split(text, NullDictionary())[1] == []  # the dictionary plate: no slots, still lossless


def test_niqqud_overlay_round_trips(words: WordList) -> None:
    real = OverlayTokenizer(LANG)
    text = "בְּרֵאשִׁית בָּרָא, ׳שלום׳ abc 123\r\n"
    template, ids = real.split(text, words)
    assert 0xFE in template and real.join(template, [words.word(i) for i in ids]) == text


def test_damaged_template_is_a_typed_error(words: WordList) -> None:
    real = OverlayTokenizer(LANG)
    template, ids = real.split("שלום עולם", words)
    names = [words.word(i) for i in ids]
    for bad_template, bad_words in ((template, names[:1]), (template, [*names, "x"]), (b"\xfe\x01", ["a"])):
        with pytest.raises(CorruptPackage):
            real.join(bad_template, bad_words)
    with pytest.raises(CorruptPackage):
        LiteralTokenizer().join(b"abc", ["x"])
