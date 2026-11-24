"""Rows -> the table the page shows, and the one-line summary. Pure arithmetic, no I/O.

Honesty rules: "identical" is only ever set after a real decompress and
byte compare; a mismatch is shown loudly and is never counted as a win; if our tool is not
the smallest, the summary says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atar.domain.extract import ltr


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one tool did with the text. ``size`` is None when it did not finish."""

    tool_id: str
    label: str
    ours: bool
    size: int | None
    compress_s: float | None
    decompress_s: float | None
    identical: bool | None
    status: str  # "ok" | "mismatch" | a Hebrew failure word ("נכשל", "לא רץ (...)")

    @property
    def counts(self) -> bool:
        """Only a finished, byte-identical package takes part in the ranking."""
        return self.status == "ok" and self.size is not None and self.identical is True


def _pct(part: int, whole: int) -> float:
    return round((whole - part) * 100.0 / whole, 1) if whole else 0.0


def table(outcomes: list[Outcome], original: int) -> list[dict[str, Any]]:
    """Rows for the page: counted rows smallest first, then the rest in run order."""
    zip_size = next((o.size for o in outcomes if o.tool_id == "zip" and o.counts), None)
    ranked = sorted((o for o in outcomes if o.counts), key=lambda o: (o.size or 0, o.label))
    rest = [o for o in outcomes if not o.counts]
    rows = []
    for o in ranked + rest:
        rows.append({
            "id": o.tool_id, "label": o.label, "ours": o.ours, "status": o.status,
            "size": o.size, "compress_s": o.compress_s, "decompress_s": o.decompress_s,
            "identical": o.identical,
            "saved_pct": _pct(o.size, original) if o.size is not None else None,
            "vs_zip_pct": _pct(o.size, zip_size) if o.size is not None and zip_size else None,
        })
    return rows


def summary(outcomes: list[Outcome], finished: bool) -> str:
    """One plain-Hebrew sentence computed from the real numbers."""
    bad_ours = [o.label for o in outcomes if o.ours and o.status == "mismatch"]
    if bad_ours:
        return f"אזהרה: {', '.join(ltr(b) for b in bad_ours)} לא נפתח זהה למקור — התוצאה שלו לא נספרת."
    ours = [o for o in outcomes if o.ours and o.counts]
    others = [o for o in outcomes if not o.ours and o.counts]
    if not ours or not others:
        return "ממתין לתוצאות..." if not finished else "אין מספיק תוצאות להשוואה."
    best = min(ours, key=lambda o: o.size or 0)
    rival = min(others, key=lambda o: o.size or 0)
    b, r = best.size or 0, rival.size or 0
    if b < r:
        return (f"הכלי שלנו ({ltr(best.label)}) קטן ב-{_pct(b, r)}% "
                f"מהמתחרה הטוב ביותר ({ltr(rival.label)}).")
    if b == r:
        return f"תיקו: הכלי שלנו ({ltr(best.label)}) ו-{ltr(rival.label)} יצאו באותו גודל בדיוק."
    return (f"הכלי שלנו לא היה הקטן ביותר בקובץ הזה: {ltr(rival.label)} קטן ב-{_pct(r, b)}% "
            f"מ-{ltr(best.label)}.")
