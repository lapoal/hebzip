"""Package -> original bytes: header, signature check, checksum, drawers.

Order matters: the format signature is compared BEFORE any drawer or model is loaded, so a package
made with other rows or tables is refused (``TableMismatch``) without decoding garbage; the checksum
(if the package carries one) is compared before the math runs. Damage inside the decoder ends in a
typed ``CorruptPackage`` / ``DamagedStream`` with bounded work (design decision). Old raw packages
of the research prototype (no header) are not read here any more (design decision).
"""

from __future__ import annotations

from hebzip.core.errors import ChecksumMismatch, CorruptPackage, TableMismatch
from hebzip.core.signature import short
from hebzip.domain.header import Header, kind_of, parse_header
from hebzip.services.select import Assembly


def decompress_bytes(asm: Assembly, data: bytes) -> bytes:
    """A hebzip v1 package -> the original bytes."""
    if kind_of(data) != "v1":
        raise CorruptPackage("זו לא חבילת hebzip")
    header = parse_header(data)
    mode = asm.mode_by_id(header.mode_id)
    if mode is None:
        raise CorruptPackage(f"מצב לא מוכר (id {header.mode_id})")
    local = asm.signature(mode)
    if local[:len(header.table_sig)] != header.table_sig:
        found, wanted = header.table_sig.hex(), short(local.hex(), 2 * len(header.table_sig))
        raise TableMismatch(f"{mode}: החבילה נוצרה עם הגדרות או טבלאות אחרות (חתימה {found} מול {wanted})")
    payload = data[header.payload_at:]
    _check_checksum(asm, header, payload)
    return asm.payload(mode).decode(payload, asm.settings.hebzip.max_output_bytes)


def _check_checksum(asm: Assembly, header: Header, payload: bytes) -> None:
    drawer = asm.integrity_by_code(header.integrity)
    if drawer.compute(payload) != header.checksum:
        raise ChecksumMismatch("החבילה פגומה: סכום הביקורת לא תואם")
