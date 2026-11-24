"""The word coder ``direct`` (mode fast, and the plate of the family).

Why a drawer: mode max codes the same table letter by letter (``domain/spell``); fast
codes it in ONE range-coder step. Taking that step out of the table into its own drawer lets the
word model stay one skeleton while the way a word reaches the coder is one word in the config.
"""

from __future__ import annotations

from hebzip.domain.contracts import CoderPort, CodingTablePort


class DirectWordCoder:
    """One symbol over the whole word table (exactly what ``WordTable.code`` did inside the table)."""

    name = "direct"

    def start(self) -> None:
        """Nothing to forget."""

    def code(self, coder: CoderPort, table: CodingTablePort, x: int) -> int:
        """Encode ``x`` / decode one word id in one step."""
        return table.code(coder, x)
