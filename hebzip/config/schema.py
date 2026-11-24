"""The typed shape of config/hebzip.toml and the vocabulary of its words.

Why frozen dataclasses: ``core.shapes`` builds them straight from the TOML dict and refuses any
unknown field at any level - a typo must refuse to start, not silently fall back (the layer table).
Why every number of the heart is ``X | None``: which rows a drawer needs depends on its word
(``kind``); ``validate_model.py`` demands exactly those rows for that word and refuses the others,
so a missing row is an error at start-up, never a hidden default inside the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

FORMAT_VERSIONS: Final = (1,)
FALLBACK_POLICIES: Final = ("smaller", "on_error")
HOLES_WORDS: Final = ("null",)
INTEGRITY_WORDS: Final = ("none", "crc8", "crc16")
ENGINE_WORDS: Final = ("python", "numba")
STORE_WORDS: Final = ("eager", "mmap", "null")
"""``[hebzip].table_store``: the pinned files read whole, their mapped derived copy, the plate."""
NARROW_WIDTHS: Final = (8, 16, 32, 64)
"""The integer widths ``[hebzip].narrow_bits`` may allow (bits)."""
LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR")
KINDS: Final = ("model", "generic", "choose")
"""``choose``: compress with each of its ``candidates`` and keep the smallest package; the
header carries the chosen candidate's id, so a choose mode never appears in a package."""
FALLBACK_KINDS: Final[dict[str, bool]] = {"model": False, "generic": True, "choose": False}
CHOOSE_KINDS: Final[dict[str, bool]] = {"model": True, "generic": True, "choose": False}
"""Mode kind -> may it be a candidate of a ``choose`` mode (a choose of chooses is refused)."""
"""Mode kind -> may it be named in ``[hebzip].fallback`` (a row table, not an if)."""
CHAIN_CONTEXTS: Final = ("cross", "within")
"""Letter-chain context words (``[modes.m.letter_chains.<n>].context``)."""

MODE_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "dictionary": ("word_list", "null"),
    "tokenizer": ("overlay", "literal"),
    "word_model": ("mixture", "null"),
    "word_coder": ("direct", "spelled"),
    "generic_codec": ("best_of", "null"),
}
"""Drawer words chosen directly on ``[modes.<m>]``."""

PLATE: Final = {"dictionary": "null", "tokenizer": "literal", "word_model": "null", "generic_codec": "null",
                "template": "flat", "base": "uniform", "mixer": "static", "predictor": "null",
                "word_coder": "direct", "gap": "none", "letter": "null", "letter_mixer": "word_only"}
"""A missing word means the plate (the plate-by-default rule)."""

TEMPLATE_NUMBERS: Final = ("prob_bits", "fix_bits", "word_distance_cap", "word_seed_order", "static_scale",
                           "conf_level", "learning_rate", "count_limit", "adaptive_start", "adaptive_min",
                           "adaptive_max", "adaptive_recip_offset2",
                           "bias_input", "weight_max", "hash_mul", "c0_mul", "seed_mul", "seed_add", "finish_mul",
                           "finish_shift1", "finish_shift2")
KIND_ROWS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "template": {"cm": ("table", "weights", "orders", "adaptive", "word_context", "squash_knots") + TEMPLATE_NUMBERS,
                 "flat": ()},
    "base": {"unigram_zipf": ("table", "zeta", "u_scale_bits", "zipf_weight_bits", "zipf_offset"), "uniform": ()},
    "mixer": {"trust": ("eta", "alpha"),
              "trust_selected": ("eta", "alpha", "trust_sets", "select_channel", "position_split"), "static": ()},
    "predictor": {"corpus_pairs": ("table", "discount", "trust0"), "text_pairs": ("confidence_k", "trust0"),
                  "text_counts": ("confidence_k", "trust0"), "text_triples": ("confidence_k", "trust0"),
                  "text_gap_starts": ("confidence_k", "trust0", "groups"),
                  "corpus_gap_starts": ("table", "discount", "trust0", "groups"),
                  "prefix_class": ("table", "prior_table", "confidence_k", "prior_k", "prior_add", "trust0"),
                  "recency": ("window", "confidence_k", "trust0"),
                  "corpus_classes": ("table", "prior_table", "window", "discount", "trust0"),
                  "text_topic": ("table", "window", "confidence_k", "trust0"), "null": ()},
    "gap": {"punctuation": ("groups",), "none": ()},
    "letter": {"static_chain": ("chain",), "text_letters": ("base_chain", "max_order", "gamma"), "null": ()},
    "letter_mixer": {"geometric": ("lr", "start_weights", "depth_max", "conf_buckets", "start_sets"),
                     "word_only": ()},
}
"""Drawer kind -> the rows it needs (and the only rows it may have). A plate ignores every row.
The tables a drawer reads follow from it: its rows named in ``TABLE_ROWS`` (``config/drawers.py``)."""
TABLE_ROWS: Final = ("table", "weights", "prior_table", "start_sets", "trust_sets")
"""Drawer rows that name a ``[tables.*]``."""

CODEC_FORMS: Final[dict[str, str]] = {
    "raw": r"raw", "deflate": r"deflate:[1-9]", "bzip2": r"bzip2:[1-9]", "xz": r"xz:[0-9](:extreme)?",
    "zstd": r"zstd:([1-9]|1[0-9]|2[0-2])", "ppmd": r"ppmd:[HI]:([2-9]|[1-5][0-9]|6[0-4])"}
"""Generic codec family -> the whole word it accepts (``family[:level][:option]``); the family is a
key of the codec registry in ``integrations/generic.py`` (resolved by the selector)."""
CODEC_OPTIONS: Final = ("extreme", "H", "I")
"""Option words inside codec words (xz's extreme preset, ppmd's variants)."""
CODEC_ROWS: Final[dict[str, tuple[str, ...]]] = {"ppmd": ("ppmd_memory_mb",)}
"""Codec family -> the mode rows it needs."""

RECIPE_ROWS: Final[dict[str, tuple[str, ...]]] = {
    "corpus_words": ("held_every",),
    "word_pairs": ("held_every",),
    "gap_starts": ("held_every", "keep_top"),
    "template_prior": ("held_every", "min_count", "count_levels"),
    "template_weights": ("held_every", "fit_cut", "fit_passes", "start_weight"),
    "prefix_classes": ("short_len", "double_min_len", "double_first"),
    "prefix_prior": ("held_every",),
    "letter_trie": ("columns",),
    "letter_chains": ("held_every",),
    "letter_start_sets": (),
    "word_classes": ("classes", "fit_words", "iterations", "seed"),
    "class_pairs": ("held_every", "min_count"),
    "trust_start_sets": (),
    "spread_sets": ("source", "halve"),
    "topic_neighbours": ("neighbours", "skip_top", "candidates", "sim_floor", "levels"),
    "append_sets": ("source", "append"),
}
"""``[tables.<name>.recipe].kind`` -> its rows: how ``ops/build_langpack`` makes the table from a corpus.
A table without a recipe is given (copied in), and the builder only reads it."""


@dataclass(frozen=True)
class RecipeRow:
    """``[tables.<name>.recipe]`` - how the table is built from a corpus (rows per kind: ``RECIPE_ROWS``)."""

    kind: str
    held_every: int | None = None
    keep_top: int | None = None
    min_count: int | None = None
    count_levels: list[int] | None = None
    fit_cut: int | None = None
    fit_passes: int | None = None
    start_weight: int | None = None
    short_len: int | None = None
    double_min_len: int | None = None
    double_first: str | None = None
    columns: int | None = None
    classes: int | None = None
    fit_words: int | None = None
    iterations: int | None = None
    seed: int | None = None
    source: str | None = None
    halve: list[int] | None = None
    neighbours: int | None = None
    """``topic_neighbours``: the most similar words kept per word ..."""
    skip_top: int | None = None
    """... among the word ids ``skip_top`` .. ``candidates - 1`` (the word list is in frequency order: the most
    frequent words - function words - neither bring a topic nor get one; rarer words than ``candidates`` neither)"""
    candidates: int | None = None
    sim_floor: float | None = None
    """cosine below this = not a neighbour; above it the weight rises linearly to ``levels`` at cosine 1."""
    levels: int | None = None
    append: list[float] | None = None
    """``append_sets``: start trust of channels added at the end of every vector of ``source``."""


@dataclass(frozen=True)
class TableFile:
    """One table file: path (relative to the project root, or absolute), sha256, size, and its recipe."""

    path: str
    sha256: str
    bytes: int
    recipe: RecipeRow | None = None


@dataclass(frozen=True)
class LanguageRow:
    """``[languages.<name>]`` - the language pack (design decision)."""

    letters: list[list[int]]
    marks: list[list[int]]
    word_list: str
    slot_plain: int
    slot_marked: int
    mark_separator: int
    marked_end: int
    sentence_end: list[str] = field(default_factory=list)
    prefixes: str = ""
    gap_space: str = ""
    gap_marks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemplateRow:
    """``[modes.<m>.template]`` - kind ``cm`` needs every row below, ``flat`` none."""

    kind: str | None = None
    table: str | None = None
    weights: str | None = None
    orders: list[int] | None = None
    adaptive: list[int] | None = None
    word_context: bool | None = None
    prob_bits: int | None = None
    fix_bits: int | None = None
    word_distance_cap: int | None = None
    word_seed_order: int | None = None
    static_scale: int | None = None
    conf_level: int | None = None
    learning_rate: int | None = None
    count_limit: int | None = None
    adaptive_start: int | None = None
    adaptive_min: int | None = None
    adaptive_max: int | None = None
    adaptive_recip_offset2: int | None = None
    bias_input: int | None = None
    weight_max: int | None = None
    hash_mul: int | None = None
    c0_mul: int | None = None
    seed_mul: int | None = None
    seed_add: int | None = None
    finish_mul: int | None = None
    finish_shift1: int | None = None
    finish_shift2: int | None = None
    squash_knots: list[int] | None = None


@dataclass(frozen=True)
class BaseRow:
    """``[modes.<m>.base]`` - kind ``unigram_zipf`` or the plate ``uniform``."""

    kind: str | None = None
    table: str | None = None
    zeta: float | None = None
    u_scale_bits: int | None = None
    zipf_weight_bits: int | None = None
    zipf_offset: int | None = None


@dataclass(frozen=True)
class MixerRow:
    """``[modes.<m>.mixer]`` - kind ``trust`` or the plate ``static``."""

    kind: str | None = None
    eta: float | None = None
    alpha: float | None = None
    trust_sets: str | None = None
    select_channel: str | None = None
    position_split: int | None = None


@dataclass(frozen=True)
class PredictorRow:
    """``[modes.<m>.predictors.<name>]`` - one mixer channel; kind ``null`` = the plate."""

    kind: str | None = None
    table: str | None = None
    discount: float | None = None
    confidence_k: float | None = None
    trust0: float | None = None
    groups: list[int] | None = None
    window: int | None = None
    prior_table: str | None = None
    prior_k: float | None = None
    prior_add: float | None = None


@dataclass(frozen=True)
class GapRow:
    """``[modes.<m>.gap]`` - kind ``punctuation`` (classes from the language rows) or the plate ``none``."""

    kind: str | None = None
    groups: list[int] | None = None


@dataclass(frozen=True)
class SpellerRow:
    """``[modes.<m>.speller]`` - the rows of word_coder ``spelled``."""

    trie: str
    q_bits: int
    inputs: list[str]


@dataclass(frozen=True)
class ChainRow:
    """``[modes.<m>.letter_chains.<name>]`` - one static letter chain (a table + its rows, not a drawer)."""

    context: str
    table: str
    max_order: int
    beta: float
    add_count: float
    symbol_bits: int


@dataclass(frozen=True)
class LetterRow:
    """``[modes.<m>.letters.<name>]`` - one letter input of the speller; kind ``null`` = the plate."""

    kind: str | None = None
    chain: str | None = None
    base_chain: str | None = None
    max_order: int | None = None
    gamma: float | None = None


@dataclass(frozen=True)
class LetterMixerRow:
    """``[modes.<m>.letter_mixer]`` - kind ``geometric`` or the plate ``word_only``."""

    kind: str | None = None
    lr: float | None = None
    start_weights: list[float] | None = None
    depth_max: int | None = None
    conf_buckets: list[float] | None = None
    start_sets: str | None = None


@dataclass(frozen=True)
class ModeRow:
    """``[modes.<name>]``: id, kind, drawer words and the sub-tables of a model mode."""

    id: int
    kind: str
    retired: bool = False
    language: str | None = None
    dictionary: str | None = None
    tokenizer: str | None = None
    word_model: str | None = None
    table_scale_bits: int | None = None
    channels: list[str] = field(default_factory=list)
    template: TemplateRow | None = None
    base: BaseRow | None = None
    mixer: MixerRow | None = None
    predictors: dict[str, PredictorRow] = field(default_factory=dict)
    word_coder: str | None = None
    gap: GapRow | None = None
    speller: SpellerRow | None = None
    letter_chains: dict[str, ChainRow] = field(default_factory=dict)
    letters: dict[str, LetterRow] = field(default_factory=dict)
    letter_mixer: LetterMixerRow | None = None
    generic_codec: str | None = None
    codecs: list[str] = field(default_factory=list)
    ppmd_memory_mb: int | None = None
    candidates: list[str] | None = None


@dataclass(frozen=True)
class HebzipRow:
    """``[hebzip]`` - the global rows."""

    format_version: int
    default_mode: str
    fallback: list[str]
    fallback_policy: str
    verify_roundtrip: bool
    integrity: str
    engine: str
    holes: str
    signature_cache: str
    material_dir: str
    """the machine-and-material rule: the ONE folder that may hold text. Nothing else in the
    project may - the canary gate scans everything outside it after a real run."""
    table_store: str
    """Where the tables come from - a word of ``STORE_WORDS`` (not in the signature: same bytes)."""
    table_store_dir: str
    """The derived (mapped) copy and its manifest, relative to the project (``ops/convert_tables`` writes it)."""
    narrow_bits: list[int]
    """The integer widths the derived copy may use, ascending (the converter refuses a value that fits none)."""
    max_input_bytes: int
    max_output_bytes: int
    phantom_bits_max: int
    log_level: str


@dataclass(frozen=True)
class Settings:
    """The whole file."""

    hebzip: HebzipRow
    tables: dict[str, TableFile]
    languages: dict[str, LanguageRow]
    modes: dict[str, ModeRow]
