"""The table-store socket: contract, plate, narrowing and the value hash.

Why: the tables are half of the format, so a store that hands out other values - a wrapped integer,
a cut width - would silently change packages. Three promises are proven here in-process; the real
converter and the CLI in both store modes are proven in ``tests/ops/test_convert_tables.py``:
  * every store (``eager``, ``mmap``, the plate ``null``) answers the same contract;
  * the plate refuses every table (``TableMismatch``) - the safe direction: nothing is invented;
  * narrowing picks the narrowest allowed width of the SAME signedness that holds every value, and a
    value that fits no allowed width is a typed error (``NarrowingOverflow``), never a truncation.
"""

from __future__ import annotations

import tomllib
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from hebzip.config.load import load_config
from hebzip.core.errors import NarrowingOverflow, TableMismatch
from hebzip.services.select import table_refs
from hebzip.store.eager import EagerTableStore
from hebzip.store.manifest import values_sha256
from hebzip.store.mapped import MmapTableStore
from hebzip.store.narrow import narrow, narrowest
from hebzip.store.port import NullTableStore, TableStorePort
from tests.support import CONFIG, ROOT, STORE_DIR_ROW, STORE_ROW, assert_same_signatures, replaced, store_dir, store_row

ALL_BITS = [8, 16, 32, 64]


def test_every_store_answers_the_same_contract() -> None:
    stores: list[Any] = [EagerTableStore({}, None), MmapTableStore({}, ROOT / "no_such_folder", None),
                         NullTableStore()]
    for store in stores:
        assert isinstance(store, TableStorePort)
    assert_same_signatures(TableStorePort, *stores)
    assert [s.name for s in stores] == ["eager", "mmap", "null"]


def test_the_plate_refuses_every_table_and_verifies_nothing() -> None:
    plate = NullTableStore()
    assert plate.verify([]) == {}
    with pytest.raises(TableMismatch):
        plate.verify(["he_words"])
    for ask in (lambda: plate.arrays("x"), lambda: plate.array("x"), lambda: plate.words("x"),
                lambda: plate.matrix("x", "weights"), lambda: plate.weight_sets("x", "wsets")):
        with pytest.raises(TableMismatch):
            ask()


def test_a_missing_derived_copy_is_refused_naming_the_converter() -> None:
    store = MmapTableStore({}, ROOT / "tests" / "_work" / "no_such_copy", None)
    with pytest.raises(TableMismatch, match="convert_tables"):
        store.verify(["he_words"])


@pytest.mark.parametrize(("values", "dtype", "wanted"), [
    ([0, 100], np.int64, np.int8),
    ([-200, 5], np.int64, np.int16),
    ([0, 70_000], np.int64, np.int32),
    ([0, 1 << 40], np.int64, np.int64),
    ([0, 70_000], np.uint32, np.uint32),
    ([0, 9_282], np.uint32, np.uint16),
    ([0, 27], np.uint8, np.uint8),
    ([], np.int64, np.int8),
])
def test_the_narrowest_width_of_the_same_signedness_is_chosen(values: list[int], dtype: type, wanted: type) -> None:
    source: npt.NDArray[Any] = np.array(values, dtype=dtype)
    assert narrowest(source, ALL_BITS) == np.dtype(wanted)
    out = narrow(source, ALL_BITS)
    assert out.dtype == np.dtype(wanted) and np.array_equal(out, source) and out.shape == source.shape


def test_non_integer_arrays_are_kept_as_they_are() -> None:
    sources: list[npt.NDArray[Any]] = [np.array([0.5, 1.25]), np.array(["a", "b"])]
    for source in sources:
        assert narrow(source, ALL_BITS).dtype == source.dtype


@pytest.mark.parametrize(("values", "bits"), [([0, 70_000], [8, 16]), ([0, 1 << 40], [8, 16, 32]),
                                               ([-129, 0], [8]), ([0, 256], [8])])
def test_a_value_that_fits_no_allowed_width_is_refused_never_cut(values: list[int], bits: list[int]) -> None:
    source = np.array(values, dtype=np.int64)
    with pytest.raises(NarrowingOverflow) as caught:
        narrow(source, bits)
    assert isinstance(caught.value, TableMismatch) and caught.value.exit_code == 3


def test_the_value_hash_sees_values_and_shape_not_width() -> None:
    wide = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert values_sha256(wide) == values_sha256(wide.astype(np.int8))
    assert values_sha256(wide) != values_sha256(wide.reshape(4))
    changed = wide.copy()
    changed[1, 1] = 5
    assert values_sha256(wide) != values_sha256(changed)
    assert values_sha256(np.array([1], dtype=np.uint8)) != values_sha256(np.array([1], dtype=np.int8))


def test_the_eager_store_reads_the_pinned_files_it_was_given() -> None:
    settings = load_config(CONFIG)
    store = EagerTableStore(table_refs(settings, ROOT), None)
    shas = store.verify(["template_m128_weights", "prefix_prior"])
    assert shas == {n: settings.tables[n].sha256 for n in ("template_m128_weights", "prefix_prior")}
    assert store.matrix("template_m128_weights", "weights").shape == (16, 15)
    assert store.array("prefix_prior").shape == (17, 3, 16)


@pytest.mark.parametrize("real", ["eager", "mmap"])
def test_a_store_variant_is_made_from_whatever_store_the_real_config_selects(real: str) -> None:
    """The variant helper replaced the hard-coded text 'table_store = "eager"',
    so the user's switch of the real config to mmap broke every store variant. Here the real config is
    taken with EACH real choice, and from each every store word must come out - with the derived copy
    moved out of the real ``tables/mmap``."""
    text = replaced(CONFIG.read_text(encoding="utf-8"), (STORE_ROW, f'table_store = "{real}"'))
    for store in ("eager", "mmap", "null"):
        variant = tomllib.loads(replaced(text, (store_row(text), f'table_store = "{store}"'),
                                         (STORE_DIR_ROW, f'table_store_dir = "{store_dir("v")}"')))
        assert variant["hebzip"]["table_store"] == store
        assert variant["hebzip"]["table_store_dir"] == "tests/_work/v/mm"
