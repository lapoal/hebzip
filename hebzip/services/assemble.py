"""Builders: config rows + table files -> the domain's drawers (one function per real drawer).

Why separate from the selector: ``select.py`` decides WHICH drawer a word means (one table);
these functions only know HOW to build each one from its rows. Tables are read once per process
(the corpus table serves the base and the corpus pairs) after their sha256 was checked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hebzip.config.schema import BaseRow, LanguageRow, MixerRow, PredictorRow, TemplateRow
from hebzip.core.errors import TableMismatch
from hebzip.domain.contracts import KernelSet
from hebzip.domain.dictionary import WordList
from hebzip.domain.lang import LanguagePack
from hebzip.domain.template_cm import CmSpec, CmTemplateModel
from hebzip.domain.template_tables import Precision, static_tables
from hebzip.domain.words.base import UnigramZipfBase
from hebzip.domain.words.mixer import TrustMixer
from hebzip.domain.words.mixer_selected import SelectedTrustMixer
from hebzip.domain.words.predictors import CorpusPairs, TextCounts, TextPairs
from hebzip.store.port import Arrays, TableStorePort, WeightSets

IArr = npt.NDArray[np.int64]
Ints = npt.NDArray[Any]
"""An integer array of any width (checked by ``Loader.ints``)."""
WEIGHTS_KEY = "weights"
PAIR_ARRAYS = ("ctx", "ctx_n", "off", "nxt_d", "cnt")
UNIGRAM_ARRAY = "uni"


class Loader:
    """Keeps the tables of one process, read once from the table store (pinned files or mapped copy)."""

    def __init__(self, store: TableStorePort) -> None:
        self.store = store
        self._arrays: dict[str, Arrays] = {}
        self._words: dict[str, list[str]] = {}
        self._array: dict[str, npt.NDArray[np.generic]] = {}

    def arrays(self, name: str) -> Arrays:
        """All arrays of an ``.npz`` table (read once)."""
        if name not in self._arrays:
            self._arrays[name] = self.store.arrays(name)
        return self._arrays[name]

    def array(self, name: str) -> npt.NDArray[np.generic]:
        """A one-array ``.npy`` table (read once)."""
        if name not in self._array:
            self._array[name] = self.store.array(name)
        return self._array[name]

    def words(self, name: str) -> list[str]:
        """A word-list table (read once)."""
        if name not in self._words:
            self._words[name] = self.store.words(name)
        return self._words[name]

    def matrix(self, name: str, key: str) -> IArr:
        """A JSON integer-matrix table."""
        return self.store.matrix(name, key)

    def weight_sets(self, name: str, key: str) -> WeightSets:
        """A JSON weight-set table."""
        return self.store.weight_sets(name, key)

    def int64(self, name: str, key: str) -> IArr:
        """One integer array of a table as an int64 COPY; a missing array = the table is not what the row says."""
        return self.ints(name, key).astype(np.int64)

    def ints(self, name: str, key: str) -> Ints:
        """One integer array as it is stored - pinned width or narrowed, maybe a read-only map - for drawers
        that never let the width reach their numbers (no int64 copy of a mapped file)."""
        try:
            array = self.arrays(name)[key]
        except KeyError as error:
            raise TableMismatch(f"בטבלה {name} חסר המערך {key}") from error
        if not np.issubdtype(array.dtype, np.integer):
            raise TableMismatch(f"בטבלה {name} המערך {key} אינו של מספרים שלמים ({array.dtype})")
        return array


def language(row: LanguageRow, name: str) -> LanguagePack:
    """The language pack from ``[languages.<name>]``."""
    return LanguagePack(name, tuple((a, b) for a, b in row.letters), tuple((a, b) for a, b in row.marks),
                        row.slot_plain, row.slot_marked, row.mark_separator, row.marked_end,
                        tuple(row.sentence_end), row.prefixes, row.gap_space, tuple(row.gap_marks))


def word_list(loader: Loader, lang: LanguageRow) -> WordList:
    """The dictionary drawer ``word_list``."""
    return WordList(loader.words(lang.word_list))


def cm_spec(row: TemplateRow) -> CmSpec:
    """The numbers of a ``cm`` template row (every row present - checked at start-up)."""
    assert row.squash_knots is not None
    hashes = (row.hash_mul, row.c0_mul, row.seed_mul, row.seed_add, row.finish_mul, row.finish_shift1,
              row.finish_shift2)
    learn = (row.learning_rate, row.count_limit, row.adaptive_start, row.adaptive_min, row.adaptive_max,
             row.bias_input, row.word_distance_cap, row.word_seed_order, row.weight_max)
    return CmSpec(tuple(row.orders or ()), tuple(row.adaptive or ()), bool(row.word_context),
                  tuple(int(v or 0) for v in hashes), tuple(int(v or 0) for v in learn),
                  tuple(row.squash_knots), int(row.adaptive_recip_offset2 or 0),
                  Precision(int(row.prob_bits or 0), int(row.fix_bits or 0)))


def cm_model(row: TemplateRow, static: Arrays, weights: IArr, kernels: KernelSet) -> CmTemplateModel:
    """A ``cm`` template model from its rows, its static-table arrays and its start weights (shared with
    the table builder, which fits the start weights on arrays it has just built)."""
    spec = cm_spec(row)
    tables = static_tables(static, spec.ctx_names(), int(row.conf_level or 0), int(row.static_scale or 0))
    return CmTemplateModel(spec, tables, weights, kernels)


def cm_template(row: TemplateRow, loader: Loader, kernels: KernelSet) -> CmTemplateModel:
    """The template drawer ``cm`` (every row present - checked at start-up)."""
    assert row.table is not None and row.weights is not None
    return cm_model(row, loader.arrays(row.table), loader.matrix(row.weights, WEIGHTS_KEY), kernels)


def unigram_zipf(row: BaseRow, loader: Loader, size: int) -> UnigramZipfBase:
    """The base drawer ``unigram_zipf``; its id count must equal the dictionary's."""
    assert row.table is not None and row.zeta is not None
    uni = loader.arrays(row.table).get(UNIGRAM_ARRAY)
    if uni is None or len(uni) != size:
        raise TableMismatch(f"טבלת הבסיס {row.table}: מספר המילים שונה מהמילון ({size})")
    return UnigramZipfBase(uni, row.zeta, int(row.u_scale_bits or 0), int(row.zipf_weight_bits or 0),
                           int(row.zipf_offset or 0))


def corpus_pairs(row: PredictorRow, loader: Loader) -> CorpusPairs:
    """The predictor ``corpus_pairs``."""
    assert row.table is not None and row.discount is not None and row.trust0 is not None
    ctx, ctx_n, off, nxt_d, cnt = (loader.ints(row.table, key) for key in PAIR_ARRAYS)
    return CorpusPairs(ctx, ctx_n, off, nxt_d, cnt, row.discount, row.trust0)


def text_pairs(row: PredictorRow, loader: Loader) -> TextPairs:
    """The predictor ``text_pairs``."""
    assert row.confidence_k is not None and row.trust0 is not None
    return TextPairs(row.confidence_k, row.trust0)


def text_counts(row: PredictorRow, loader: Loader) -> TextCounts:
    """The predictor ``text_counts``."""
    assert row.confidence_k is not None and row.trust0 is not None
    return TextCounts(row.confidence_k, row.trust0)


SITUATIONS = 4
"""Per gap class, ``trust_selected`` tells four situations apart: (found a corpus list) x (n >= position_split)."""
SETS_KEY = "wsets"
"""The JSON key of a start-set table: the letter mixer's ``start_sets`` {"<depth>,<confidence bucket>": [...]}
and the ``trust_sets`` of ``trust_selected`` {"<gap class>,<found * 2 + late>": [one trust per channel]} -
one layout, so one recipe (``spread_sets``) serves both."""


def selected_mixer(row: MixerRow, loader: Loader, channels: list[str], active: list[str]) -> SelectedTrustMixer:
    """The mixer ``trust_selected``: one pretrained trust per CONFIGURED channel (``channels`` row order);
    the mixer sees only the active ones (a channel switched to its plate is left out of the mixer, so its
    trust is dropped here - the rest keep theirs). Its select channel switched off = "never found"."""
    assert row.eta is not None and row.alpha is not None and row.trust_sets is not None
    assert row.select_channel is not None and row.position_split is not None
    raw = loader.weight_sets(row.trust_sets, SETS_KEY)
    if any(len(v) != len(channels) or not 0 <= s <= SITUATIONS - 1 for (_, s), v in raw.items()):
        raise TableMismatch(f"טבלת {row.trust_sets}: צריך וקטור אחד לכל ערוץ ({len(channels)}) ומצב 0..3")
    keep = [channels.index(n) for n in active]
    sets = {g * SITUATIONS + s: [v[i] for i in keep] for (g, s), v in raw.items()}
    at = active.index(row.select_channel) if row.select_channel in active else -1
    return SelectedTrustMixer(row.eta, row.alpha, sets, at, row.position_split)


def trust_mixer(row: MixerRow) -> TrustMixer:
    """The mixer ``trust``."""
    assert row.eta is not None and row.alpha is not None
    return TrustMixer(row.eta, row.alpha)
