"""Uploaded bytes -> the UTF-8 text every compressor receives. Pure: bytes in, bytes out.

.docx: an earlier quick regex ``<w:t[^>]*>`` also matched ``<w:tabs>``
and ``<w:tab/>`` and leaked ~4 KB of XML into the "text". So the document is parsed with a
real XML parser and only these count: ``w:t`` text, ``w:tab`` -> tab, ``w:br``/``w:cr`` ->
newline, and paragraphs (``w:p``) joined by "\\n". Everything else (properties, tab stops,
field codes in ``w:instrText``, deleted text ``w:delText``) is ignored on purpose.
"""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree

from atar.core.errors import InputRejected

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_KINDS = {"txt": "txt", "text": "txt", "docx": "docx"}


def ltr(word: str) -> str:
    """A left-to-right word isolated inside a Hebrew sentence (so ".txt" is not shown "txt.")."""
    return chr(0x2066) + word + chr(0x2069)  # LRI ... PDI, written as code points on purpose


def kind_of(file_name: str) -> str:
    """'txt' or 'docx' from the name's extension; anything else is refused."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in _TEXT_KINDS:
        raise InputRejected(f"אפשר להעלות רק קובץ {ltr('.txt')} ({ltr('UTF-8')}) או {ltr('.docx')}")
    return _TEXT_KINDS[ext]


def txt_text(data: bytes) -> str:
    """A UTF-8 text file (an optional BOM is dropped)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InputRejected("הקובץ אינו טקסט בקידוד UTF-8. שמור אותו כ-UTF-8 ונסה שוב.") from error


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Only the direct children of runs (``w:r``) count.

    Why runs only (see test_tab_stops_do_not_leak_xml): a paragraph's
    properties hold ``<w:tabs><w:tab w:pos=.../></w:tabs>`` - tab STOP definitions, also named
    ``w:tab``. Walking every node turned each tab stop into a tab character.
    """
    parts: list[str] = []
    for run in paragraph.iter(W + "r"):
        for node in run:
            if node.tag == W + "t":
                parts.append(node.text or "")
            elif node.tag == W + "tab":
                parts.append("\t")
            elif node.tag in (W + "br", W + "cr"):
                parts.append("\n")
    return "".join(parts)


def _top_paragraphs(body: ElementTree.Element) -> list[ElementTree.Element]:
    """Every w:p that is not nested inside another w:p (a text box's paragraphs stay in theirs)."""
    found: list[ElementTree.Element] = []

    def walk(node: ElementTree.Element) -> None:
        for child in node:
            if child.tag == W + "p":
                found.append(child)
            else:
                walk(child)

    walk(body)
    return found


def document_xml_text(xml: bytes) -> str:
    """The text of a ``word/document.xml`` part."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise InputRejected("קובץ ה-Word פגום (לא הצלחתי לקרוא את תוכנו)") from error
    body = root.find(W + "body")
    if body is None:
        raise InputRejected("קובץ ה-Word ריק או לא תקין")
    return "\n".join(_paragraph_text(p) for p in _top_paragraphs(body))


def docx_text(data: bytes) -> str:
    """The text of a .docx file (main document part only)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as error:
        raise InputRejected(f"זה לא נראה כמו קובץ Word ({ltr('.docx')}) תקין") from error
    return document_xml_text(xml)


def extract(file_name: str, data: bytes, max_text_bytes: int) -> bytes:
    """The UTF-8 bytes to compare, or a Hebrew refusal (empty / over the limit)."""
    text = docx_text(data) if kind_of(file_name) == "docx" else txt_text(data)
    raw = text.encode("utf-8")
    if not text.strip():
        raise InputRejected("לא נמצא טקסט בקובץ")
    if len(raw) > max_text_bytes:
        raise InputRejected(f"הטקסט גדול מדי: {len(raw) // 1024:,} KB. "
                            f"המגבלה היא {max_text_bytes // 1024:,} KB של טקסט.")
    return raw


def word_count(raw: bytes) -> int:
    """Words = runs of non-space characters."""
    return len(raw.decode("utf-8").split())
