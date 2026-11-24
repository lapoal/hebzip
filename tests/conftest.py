"""Every test cleans up after itself (the machine-and-material rule).

Why a fixture and not a line at the end of each test: a test that fails stops at its assert, and the
scratch folder with decoded copies of the golden texts would stay on disk. A fixture's teardown runs
after the test whatever happened - pass, fail or error. ``tests/run_tests.py`` then checks that
``tests/_work`` is empty after the whole suite (the second witness).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.support import clean_work


@pytest.fixture(autouse=True)
def _scratch_is_removed_after_the_test() -> Iterator[None]:
    yield
    left = clean_work()
    assert not left, f"scratch folders could not be removed: {[p.name for p in left]}"
