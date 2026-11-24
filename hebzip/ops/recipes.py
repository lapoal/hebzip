"""The table recipes: ``[tables.<name>.recipe].kind`` -> how the table is made from the corpus.

Why a registry: no if-chain on drawer words decides
which table to make or refuses a table by name. Every table says
its own recipe in the config; ``RECIPES`` maps the word to one function here, and each function takes
the rows it follows from the drawer that reads the table (``Build.reader``). A new kind of table = a
recipe word + rows in ``schema.RECIPE_ROWS`` + one function in this table; nothing else changes.
The counting itself is pure (``ops/counts.py``, ``ops/prefix.py``, ``ops/template_prior.py``,
``domain/template_fit.py``); the writers are the store's (deterministic: same corpus, same sha256).
"""

from __future__ import annotations

import numpy as np

from hebzip.config.checks import bad
from hebzip.config.schema import RecipeRow
from hebzip.core.errors import ConfigError
from hebzip.domain.template_fit import fit_weights
from hebzip.ops import classes as K
from hebzip.ops import counts as C
from hebzip.ops import prefix as X
from hebzip.ops import template_prior as T
from hebzip.ops import topics as N
from hebzip.ops.build import Build, Made, Recipe
from hebzip.ops.corpus import is_held
from hebzip.services import assemble as A
from hebzip.services.assemble import SETS_KEY, WEIGHTS_KEY
from hebzip.store.write import write_array, write_arrays, write_json, write_words


def _arrays(arrays: C.Arrays) -> Made:
    return Made(arrays, lambda path: write_arrays(path, arrays))


def _json(value: object) -> Made:
    return Made(value, lambda path: write_json(path, value))


def _int(value: int | None) -> int:
    assert value is not None  # the recipe's rows are checked at start-up (validate_recipe)
    return value


def corpus_words(b: Build, name: str, r: RecipeRow) -> Made:
    """Every word form of the training documents, most frequent first."""
    tokenizer = b.mode(b.reader(name)).tokenizer()
    words = C.ranked_words([t for n, t in b.numbered if not is_held(n, _int(r.held_every))], tokenizer)
    return Made(words, lambda path: write_words(path, words))


def word_pairs(b: Build, name: str, r: RecipeRow) -> Made:
    """Word counts (``uni``) and follower lists (``corpus_pairs``)."""
    docs = b.tokenized(b.reader(name)).part(_int(r.held_every), held=False)
    return _arrays(C.pair_arrays(docs, len(b.words(b.lang().word_list))))


def gap_starts(b: Build, name: str, r: RecipeRow) -> Made:
    """First words per gap class (the research prototype's ``cg.npz``)."""
    use = b.reader(name, agreed=True)
    build = b.mode(use)
    docs = b.tokenized(use).part(_int(r.held_every), held=False)
    return _arrays(C.gap_start_arrays(docs, build.tokenizer(), build.gap_classifier(), use.row.groups or [],
                                      _int(r.keep_top)))


def template_prior(b: Build, name: str, r: RecipeRow) -> Made:
    """The template model's static context tables (the research prototype's ``tp_tables.py``)."""
    use = b.reader(name, agreed=True)
    docs = b.tokenized(use).part(_int(r.held_every), held=False)
    numbers = (_int(r.min_count), list(r.count_levels or []), int(use.row.static_scale or 0))
    slot_end = b.mode(use).tokenizer().slot_end
    return _arrays(T.static_prior(docs, A.cm_spec(use.row), slot_end, b.kernels(), numbers))


def template_weights(b: Build, name: str, r: RecipeRow) -> Made:
    """The template mixer's start weights, fitted on every second held-out document (``tp_train.py``)."""
    use = b.reader(name, agreed=True)
    row = use.row
    spec = A.cm_spec(row)
    model = A.cm_model(row, b.arrays(row.table), T.start_weights(spec, _int(r.start_weight)), b.kernels())
    held = b.tokenized(use).part(_int(r.held_every), held=True)[0::2]
    texts = [(template[:_int(r.fit_cut)], ids) for template, ids in held]
    weights = fit_weights(model, texts, _int(r.fit_passes), b.mode(use).tokenizer().slot_end, b.kernels())
    return _json({WEIGHTS_KEY: weights.tolist()})


def prefix_classes(b: Build, name: str, r: RecipeRow) -> Made:
    """A proclitic class per word id, from the language's ``prefixes`` row (``ops/prefix.py``)."""
    lang = b.lang()
    _check_doubles(name, r, lang.prefixes)
    classes = X.class_array(b.words(lang.word_list), lang.prefixes, r.double_first or "", _int(r.short_len),
                            _int(r.double_min_len))
    return Made(classes, lambda path: write_array(path, classes))


def _check_doubles(name: str, r: RecipeRow, prefixes: str) -> None:
    stray = [ch for ch in r.double_first or "" if ch not in prefixes]
    if stray:
        raise bad("config", f"[tables.{name}.recipe].double_first", r.double_first,
                  "every letter must also be in the language's prefixes")


def prefix_prior(b: Build, name: str, r: RecipeRow) -> Made:
    """Class transitions [previous class][gap group][class] over the training documents (``sp_data.py``)."""
    use = b.reader(name)
    classes_table = use.row.table
    classes = b.array(classes_table)
    made_by = b.recipe(classes_table)
    count = (X.class_count(b.lang().prefixes, made_by.double_first or "") if made_by is not None
             else int(classes.max()) + 1)
    build = b.mode(use)
    docs = b.tokenized(use).part(_int(r.held_every), held=False)
    prior = X.prior_counts(docs, build.tokenizer(), build.gap_classifier(), classes, count)
    return Made(prior, lambda path: write_array(path, prior))


def letter_trie(b: Build, name: str, r: RecipeRow) -> Made:
    """The dictionary in alphabetical order (the speller's tree)."""
    lang = A.language(b.lang(), b.language)
    return _arrays(C.trie_arrays(b.words(b.lang().word_list), lang.alphabet(), _int(r.columns)))


def letter_chains(b: Build, name: str, r: RecipeRow) -> Made:
    """Letter-chain counts of every chain row that names this table (a symbol stream per document)."""
    use = b.reader(name)
    alphabet = A.language(b.lang(), b.language).alphabet()
    words = b.words(b.lang().word_list)
    docs = b.tokenized(use).part(_int(r.held_every), held=False)
    streams = [C.letter_stream([words[x] for x in ids], alphabet) for _, ids in docs]
    merged: C.Arrays = {}
    for chain in use.mode.letter_chains.values():
        if chain.table == name:
            merged.update(C.chain_arrays(streams, len(alphabet) + 1, chain.context, chain.max_order,
                                         chain.symbol_bits))
    return _arrays(merged)


def letter_start_sets(b: Build, name: str, r: RecipeRow) -> Made:
    """No pretrained start sets: the letter mixer starts from its ``start_weights`` row everywhere (also the
    recipe ``trust_start_sets``: every situation of ``trust_selected`` starts from the ``trust0`` rows -
    the Hebrew trust table was fitted on held-out Wikipedia by the research prototype and is given)."""
    return _json({SETS_KEY: {}})


def word_classes(b: Build, name: str, r: RecipeRow) -> Made:
    """A class per word id from the word vectors given with ``--vectors`` (``ops/classes.py``)."""
    vectors = b.vectors(name)
    size = len(b.words(b.lang().word_list))
    if vectors.ndim != 2 or len(vectors) != size:
        raise ConfigError(f"[tables.{name}]: the vectors give {vectors.shape} - need one row per word id ({size})")
    classes = K.kmeans(vectors, _int(r.classes), _int(r.fit_words), _int(r.iterations), _int(r.seed))
    return Made(classes, lambda path: write_array(path, classes))


def class_pairs(b: Build, name: str, r: RecipeRow) -> Made:
    """The corpus class lists of every ``corpus_classes`` drawer that reads this table (one per window)."""
    readers = b.readers.get(name, [])
    use = b.reader(name)
    word_class = b.array(use.row.table).astype(np.int64)
    windows = sorted({int(u.row.window) for u in readers})
    docs = b.tokenized(use).part(_int(r.held_every), held=False)
    return _arrays(K.class_lists(docs, word_class, windows, _int(r.min_count)))


def spread_sets(b: Build, name: str, r: RecipeRow) -> Made:
    """Start sets for a mode that reads a second corpus - the letter mixer's (``start_sets``) or the trust
    vectors of ``trust_selected`` (``trust_sets``; same layout): every position listed in ``halve`` shares its
    weight with its twin, appended at the end in the same order (``source`` = the sets of the one-corpus
    mode). The research prototype's ``mix_letter_sets.json``."""
    halve = list(r.halve or [])
    out: dict[str, list[float]] = {}
    for (d, c), w in sorted(b.weight_sets(r.source or "").items()):
        if any(not 0 <= i < len(w) for i in halve):
            raise ConfigError(f"[tables.{name}.recipe].halve: {halve} - the sets of {r.source} have {len(w)} weights")
        spread = [w[i] / 2 if i in halve else w[i] for i in range(len(w))]
        out[f"{d},{c}"] = spread + [w[i] / 2 for i in halve]
    return _json({SETS_KEY: out})


def topic_neighbours(b: Build, name: str, r: RecipeRow) -> Made:
    """The neighbour lists of ``text_topic`` from the word vectors given with ``--vectors`` (``ops/topics.py``)."""
    vectors = b.vectors(name)
    size = len(b.words(b.lang().word_list))
    if vectors.ndim != 2 or len(vectors) != size:
        raise ConfigError(f"[tables.{name}]: the vectors give {vectors.shape} - need one row per word id ({size})")
    assert r.sim_floor is not None  # checked at start-up (validate_recipe)
    return _arrays(N.neighbour_lists(vectors, _int(r.skip_top), _int(r.candidates), _int(r.neighbours),
                                     r.sim_floor, _int(r.levels)))


def append_sets(b: Build, name: str, r: RecipeRow) -> Made:
    """Trust start sets for a mode with channels ADDED at the end: every vector of ``source`` (same layout as
    ``spread_sets``) gets the ``append`` values after its own."""
    tail = list(r.append or [])
    out = {f"{d},{c}": list(w) + tail for (d, c), w in sorted(b.weight_sets(r.source or "").items())}
    return _json({SETS_KEY: out})


RECIPES: dict[str, Recipe] = {
    "corpus_words": corpus_words, "word_pairs": word_pairs, "gap_starts": gap_starts,
    "template_prior": template_prior, "template_weights": template_weights, "prefix_classes": prefix_classes,
    "prefix_prior": prefix_prior, "letter_trie": letter_trie, "letter_chains": letter_chains,
    "letter_start_sets": letter_start_sets, "word_classes": word_classes, "class_pairs": class_pairs,
    "trust_start_sets": letter_start_sets, "spread_sets": spread_sets, "topic_neighbours": topic_neighbours,
    "append_sets": append_sets}
"""Recipe word -> its function (every word of ``schema.RECIPE_ROWS`` is a key - tested)."""
