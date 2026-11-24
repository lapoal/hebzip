"""Text extraction: .docx through a real XML parser, .txt as UTF-8, limits and refusals."""

from __future__ import annotations

import re

import pytest

from atar.core.errors import InputRejected
from atar.domain.extract import docx_text, extract, word_count
from tests.support import HEBREW, docx_bytes, paragraph


def _old_regex_text(data: bytes) -> str:
    """The old .docx bug, kept only to prove the test below catches it."""
    import io
    import zipfile
    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    return "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.S))


def test_tab_stops_do_not_leak_xml() -> None:
    """<w:tabs>/<w:tab .../> in paragraph properties must not become text (the regex bug)."""
    data = docx_bytes(f"<w:p>{paragraph('שורה ראשונה')}</w:p><w:p>{paragraph('שורה שנייה')}</w:p>")
    text = docx_text(data)
    assert text == "שורה ראשונה\nשורה שנייה"
    assert "<" not in text and "w:" not in text and "9026" not in text


def test_old_regex_really_leaked() -> None:
    """Broken-without proof: the old regex on the same document returns something else."""
    data = docx_bytes("<w:p>" + paragraph("שורה ראשונה") + "</w:p>")
    leaked = _old_regex_text(data)
    assert "<w:" in leaked and "9026" in leaked
    assert leaked != docx_text(data)


def test_tab_break_and_paragraphs() -> None:
    body = ('<w:p><w:r><w:t>א</w:t><w:tab/><w:t>ב</w:t><w:br/><w:t>ג</w:t></w:r></w:p>'
            '<w:p><w:r><w:t xml:space="preserve"> ד </w:t></w:r></w:p><w:p/>')
    assert docx_text(docx_bytes(body)) == "א\tב\nג\n ד \n"


def test_field_codes_and_deleted_text_are_ignored() -> None:
    body = ('<w:p><w:r><w:instrText> PAGE </w:instrText></w:r>'
            '<w:del><w:r><w:delText>נמחק</w:delText></w:r></w:del>'
            '<w:r><w:t>נשאר</w:t></w:r></w:p>')
    assert docx_text(docx_bytes(body)) == "נשאר"


def test_txt_bom_dropped_and_bytes_kept() -> None:
    raw = extract("a.txt", b"\xef\xbb\xbf" + HEBREW.encode("utf-8"), 10_000)
    assert raw == HEBREW.encode("utf-8")
    assert word_count(raw) == len(HEBREW.split())


@pytest.mark.parametrize(("name", "data"), [
    ("a.pdf", b"x"), ("noext", b"x"), ("a.txt", "שלום".encode("cp1255")),
    ("a.txt", b"   \n"), ("a.docx", b"not a zip"),
])
def test_refusals_are_hebrew(name: str, data: bytes) -> None:
    with pytest.raises(InputRejected) as caught:
        extract(name, data, 10_000)
    assert re.search("[א-ת]", str(caught.value))


def test_limit_counts_extracted_text() -> None:
    text = ("מילה " * 50).encode("utf-8")
    assert extract("a.txt", text, len(text)) == text
    with pytest.raises(InputRejected, match="גדול מדי"):
        extract("a.txt", text, len(text) - 1)
