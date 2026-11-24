"""Builders of mode max's word drawers: gap classifier and the five new predictors.

Why apart from ``assemble.py``: the same split as the domain (one builder per real drawer, the
selector decides which). Every table is checked against the drawers that read it here, at build
time: a gap-start table must have a list for every gap class the classifier can produce, the class
table one class per dictionary word, the class prior one row per (previous class, gap group) -
a table that does not fit is ``TableMismatch``, never a silent wrong lookup.
"""

from __future__ import annotations

import numpy as np

from hebzip.config.schema import GapRow, PredictorRow
from hebzip.core.errors import TableMismatch
from hebzip.domain.contracts import GapClassifierPort
from hebzip.domain.gap import PunctuationGap
from hebzip.domain.lang import LanguagePack
from hebzip.domain.words.predictors_classes import ClassLists, CorpusClasses
from hebzip.domain.words.predictors_corpus import CorpusGapStarts, GapList, PrefixClass, normalised_prior
from hebzip.domain.words.predictors_text import Recency, TextGapStarts, TextTriples
from hebzip.domain.words.predictors_topic import NeighbourLists, TextTopic
from hebzip.services.assemble import Loader

GAP_LIST_ARRAYS = ("ids{}", "cnt{}", "N{}")
"""The arrays of a gap-start table per gap class g (the research prototype's ``cg.npz`` layout)."""
CLASS_LIST_ARRAYS = ("off", "ids", "cnt", "N")
"""The arrays of a class-list table per window w: ``off{w}``, ``ids{w}``, ``cnt{w}``, ``N{w}``."""
TOPIC_ARRAYS = ("off", "ids", "wt")
"""The arrays of a neighbour-list table: slice offsets per word id, neighbour ids, integer weights."""


def punctuation_gap(row: GapRow, lang: LanguagePack) -> PunctuationGap:
    """The gap classifier ``punctuation`` (classes from the language, groups from the mode row)."""
    assert row.groups is not None
    return PunctuationGap(lang.sentence_end, lang.gap_space, lang.gap_marks, tuple(row.groups))


def text_triples(row: PredictorRow, loader: Loader) -> TextTriples:
    """The predictor ``text_triples`` (T3)."""
    assert row.confidence_k is not None and row.trust0 is not None
    return TextTriples(row.confidence_k, row.trust0)


def text_gap_starts(row: PredictorRow, loader: Loader) -> TextGapStarts:
    """The predictor ``text_gap_starts`` (TG)."""
    assert row.confidence_k is not None and row.trust0 is not None and row.groups is not None
    return TextGapStarts(row.confidence_k, row.trust0, tuple(row.groups))


def recency(row: PredictorRow, loader: Loader) -> Recency:
    """The predictor ``recency`` (RC)."""
    assert row.window is not None and row.confidence_k is not None and row.trust0 is not None
    return Recency(row.window, row.confidence_k, row.trust0)


def corpus_gap_starts(row: PredictorRow, loader: Loader, gap: GapClassifierPort) -> CorpusGapStarts:
    """The predictor ``corpus_gap_starts`` (CG): one corpus list per gap class."""
    assert row.table is not None and row.discount is not None and row.trust0 is not None and row.groups is not None
    lists = []
    for g in range(gap.classes):
        ids, cnt, total = (loader.int64(row.table, name.format(g)) for name in GAP_LIST_ARRAYS)
        if len(ids) != len(cnt) or len(total) != 1 or (len(ids) > 1 and not bool(np.all(ids[1:] > ids[:-1]))):
            raise TableMismatch(f"טבלת {row.table}: רשימת סוג-רווח {g} אינה תקינה")
        lists.append(GapList(ids, cnt, int(total[0])))
    return CorpusGapStarts(lists, row.discount, row.trust0, tuple(row.groups))


def corpus_classes(row: PredictorRow, loader: Loader, size: int) -> CorpusClasses:
    """The predictor ``corpus_classes``: ``table`` = a class per word, ``prior_table`` = the corpus
    lists ``off{w}``/``ids{w}``/``cnt{w}``/``N{w}`` per class for this ``window`` (``ops/classes.py``)."""
    assert row.table is not None and row.prior_table is not None and row.window is not None
    assert row.discount is not None and row.trust0 is not None
    classes = loader.array(row.table).astype(np.int64)
    off, ids, cnt, total = (loader.int64(row.prior_table, f"{name}{row.window}") for name in CLASS_LIST_ARRAYS)
    k = len(total)
    if len(classes) != size or int(classes.min()) < 0 or int(classes.max()) >= k:
        raise TableMismatch(f"טבלת {row.table}: צריך סוג אחד (0..{k - 1}) לכל מילה במילון ({size})")
    if len(off) != k + 1 or int(off[0]) != 0 or int(off[-1]) != len(ids) or len(ids) != len(cnt) or (
            len(off) > 1 and not bool(np.all(off[1:] >= off[:-1]))):
        raise TableMismatch(f"טבלת {row.prior_table}: רשימות הסוגים (מרחק {row.window}) אינן תקינות")
    return CorpusClasses(classes, ClassLists(off, ids, cnt, total), row.window, row.discount, row.trust0)


def text_topic(row: PredictorRow, loader: Loader, size: int) -> TextTopic:
    """The predictor ``text_topic``: ``table`` = the neighbour lists ``off``/``ids``/``wt`` (``ops/topics.py``),
    one slice per dictionary word, ids sorted inside a slice, every weight >= 1. Read as stored (any integer
    width, maybe a read-only map - the drawer turns each value into a Python int)."""
    assert row.table is not None and row.window is not None and row.confidence_k is not None
    assert row.trust0 is not None
    off, ids, wt = (loader.ints(row.table, name) for name in TOPIC_ARRAYS)
    lists_ok = len(off) == size + 1 and int(off[0]) == 0 and int(off[-1]) == len(ids) == len(wt)
    if not lists_ok or bool(np.any(np.diff(off) < 0)):
        raise TableMismatch(f"טבלת {row.table}: צריך רשימת-שכנים אחת לכל מילה במילון ({size})")
    if len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= size or int(wt.min()) < 1):
        raise TableMismatch(f"טבלת {row.table}: שכן מחוץ למילון או משקל קטן מ-1")
    return TextTopic(NeighbourLists(off, ids, wt), row.window, row.confidence_k, row.trust0)


def prefix_class(row: PredictorRow, loader: Loader, size: int, gap: GapClassifierPort) -> PrefixClass:
    """The predictor ``prefix_class`` (PX): class per word + the class-transition prior."""
    assert row.table is not None and row.prior_table is not None and row.prior_add is not None
    assert row.confidence_k is not None and row.prior_k is not None and row.trust0 is not None
    classes = loader.array(row.table).astype(np.int64)
    counts = loader.array(row.prior_table)
    if counts.ndim != 3 or counts.shape[0] != counts.shape[2] + 1 or counts.shape[1] < gap.groups:
        raise TableMismatch(f"טבלת {row.prior_table}: הצורה {counts.shape} אינה [סוג-קודם+1][קבוצה][סוג]")
    if len(classes) != size or int(classes.min()) < 0 or int(classes.max()) >= counts.shape[2]:
        raise TableMismatch(f"טבלת {row.table}: צריך סוג אחד לכל מילה במילון ({size})")
    return PrefixClass(classes, normalised_prior(counts, row.prior_add), row.confidence_k, row.prior_k, row.trust0)
