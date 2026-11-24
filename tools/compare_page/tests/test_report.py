"""The table and the one-line summary are computed from real numbers, honestly."""

from __future__ import annotations

from atar.domain.extract import ltr
from atar.domain.report import Outcome, summary, table


def o(tool_id: str, size: int | None, ours: bool = False, status: str = "ok") -> Outcome:
    identical = None if size is None else status == "ok"
    return Outcome(tool_id, tool_id, ours, size, 0.1, 0.1 if size else None, identical, status)


def test_sorted_smallest_first_failures_last() -> None:
    rows = table([o("zip", 600), o("paq8px", None, status="לא רץ (אין מספיק זיכרון)"),
                  o("xz", 650), o("max", 250, ours=True)], 1500)
    assert [r["id"] for r in rows] == ["max", "zip", "xz", "paq8px"]
    assert rows[0]["saved_pct"] == round(1250 * 100 / 1500, 1)
    assert rows[0]["vs_zip_pct"] == round(350 * 100 / 600, 1)
    assert rows[2]["vs_zip_pct"] < 0
    assert rows[3]["size"] is None and rows[3]["status"].startswith("לא רץ")


def test_summary_when_we_win() -> None:
    text = summary([o("zip", 600), o("paq8px", 431), o("max", 248, ours=True)], True)
    assert text == f"הכלי שלנו ({ltr('max')}) קטן ב-42.5% מהמתחרה הטוב ביותר ({ltr('paq8px')})."


def test_summary_says_so_when_we_lose() -> None:
    text = summary([o("paq8px", 400), o("max", 500, ours=True)], True)
    assert text.startswith("הכלי שלנו לא היה הקטן ביותר")
    assert "20.0%" in text


def test_mismatch_is_loud_and_never_a_win() -> None:
    outs = [o("zip", 600), o("max", 10, ours=True, status="mismatch")]
    assert summary(outs, True).startswith("אזהרה")
    rows = table(outs, 1000)
    assert rows[-1]["id"] == "max" and rows[-1]["identical"] is False


def test_waiting_until_both_sides_have_a_result() -> None:
    assert summary([o("zip", 600)], False) == "ממתין לתוצאות..."
    assert summary([o("zip", 600)], True) == "אין מספיק תוצאות להשוואה."
