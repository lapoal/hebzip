"""Holes drawer: only the plate is built - contract + 'no holes' answer.

Any other config word is refused at start-up (``tests/config/test_load.py``)."""

from __future__ import annotations

from hebzip.domain.contracts import HolesPort
from hebzip.domain.nulls import NullHoles
from tests.support import assert_same_signatures


def test_plate_answers_the_contract_and_chooses_nothing() -> None:
    plate = NullHoles()
    assert isinstance(plate, HolesPort) and plate.name == "null"
    assert_same_signatures(HolesPort, plate)
    assert plate.choose([1, 2, 3], "context") == {}
    assert plate.choose([], "") == {}
