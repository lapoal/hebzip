"""Empty ``tests/_work`` after every test - also when the test failed (the machine-and-material rule)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator

import pytest

from tests.support import WORK


@pytest.fixture(autouse=True)
def clean_work() -> Iterator[None]:
    WORK.mkdir(parents=True, exist_ok=True)
    yield
    for child in list(WORK.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
