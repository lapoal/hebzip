"""THE selector: a word in the config table -> the real drawer or its plate.

Why one file (the one-selector rule and the choose-once-at-the-root rule): the only place that knows which
implementation a word means;
no other flow asks "which mode am I". A missing word means the plate (and a log line). Modes are
built lazily and cached per process; the tables of a mode are verified against their sha256 rows
BEFORE anything is built, and the format signature (header byte) is computed from the same rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hebzip.config.checks import bad
from hebzip.config.drawers import Use, uses
from hebzip.config.schema import PLATE, ModeRow, Settings
from hebzip.config.tables import mode_tables
from hebzip.config.validate import resolve
from hebzip.core.errors import ConfigError
from hebzip.core.logging import get_logger
from hebzip.domain.contracts import (
    GapClassifierPort,
    GenericCodecPort,
    IntegrityPort,
    KernelSet,
    LetterInputPort,
    LetterMixerPort,
    ModeParts,
    PayloadCodecPort,
    PredictorPort,
    TokenizerPort,
    WordCoderPort,
)
from hebzip.domain.dictionary import NullDictionary
from hebzip.domain.gap import NoGap, PunctuationGap
from hebzip.domain.integrity import Crc8Integrity, Crc16Integrity
from hebzip.domain.kernels_layout import AD_INPUTS_MAX
from hebzip.domain.modes import ModeSpec, format_signature
from hebzip.domain.nulls import ChoosePayload, NullGenericCodec, NullIntegrity
from hebzip.domain.payload import GenericPayload, ModelPayload
from hebzip.domain.spell.chains import LetterChain
from hebzip.domain.spell.inputs import NullLetterInput
from hebzip.domain.spell.mixer import WordOnlyMixer
from hebzip.domain.template_cm import FlatTemplateModel
from hebzip.domain.tokenizer import LiteralTokenizer, OverlayTokenizer
from hebzip.domain.words.base import UniformBase
from hebzip.domain.words.coders import DirectWordCoder
from hebzip.domain.words.mixer import StaticMixer
from hebzip.domain.words.mixture import MixtureWordModel, NullWordModel
from hebzip.domain.words.predictors import NullPredictor
from hebzip.integrations import generic as G
from hebzip.integrations.generic import BestOfCodec
from hebzip.services import assemble as A
from hebzip.services import assemble_spell as P
from hebzip.services import assemble_words as W
from hebzip.services import engines as E
from hebzip.store.eager import EagerTableStore
from hebzip.store.mapped import MmapTableStore
from hebzip.store.port import NullTableStore, TableRef, TableStorePort

LOG = get_logger("select")

REAL: dict[str, dict[str, Callable[..., Any]]] = {
    "dictionary": {"word_list": lambda b: A.word_list(b.loader, b.lang_row), "null": lambda b: NullDictionary()},
    "tokenizer": {"overlay": lambda b: OverlayTokenizer(b.lang), "literal": lambda b: LiteralTokenizer()},
    "template": {"cm": lambda b, row: A.cm_template(row, b.loader, b.k),
                 "flat": lambda b, row: FlatTemplateModel(b.k)},
    "base": {"unigram_zipf": lambda b, row, n: A.unigram_zipf(row, b.loader, n),
             "uniform": lambda b, row, n: UniformBase(n)},
    "mixer": {"trust": lambda b, row: A.trust_mixer(row),
              "trust_selected": lambda b, row: A.selected_mixer(row, b.loader, b.row.channels, b.active),
              "static": lambda b, row: StaticMixer()},
    "predictor": {"corpus_pairs": lambda b, row: A.corpus_pairs(row, b.loader),
                  "text_pairs": lambda b, row: A.text_pairs(row, b.loader),
                  "text_counts": lambda b, row: A.text_counts(row, b.loader),
                  "text_triples": lambda b, row: W.text_triples(row, b.loader),
                  "text_gap_starts": lambda b, row: W.text_gap_starts(row, b.loader),
                  "corpus_gap_starts": lambda b, row: W.corpus_gap_starts(row, b.loader, b.gap),
                  "prefix_class": lambda b, row: W.prefix_class(row, b.loader, b.size, b.gap),
                  "recency": lambda b, row: W.recency(row, b.loader),
                  "corpus_classes": lambda b, row: W.corpus_classes(row, b.loader, b.size),
                  "text_topic": lambda b, row: W.text_topic(row, b.loader, b.size),
                  "null": lambda b, row: NullPredictor()},
    "word_model": {"mixture": lambda b: b.mixture(), "null": lambda b: NullWordModel()},
    "word_coder": {"direct": lambda b: DirectWordCoder(), "spelled": lambda b: b.speller()},
    "gap": {"punctuation": lambda b, row: W.punctuation_gap(row, b.lang), "none": lambda b, row: NoGap()},
    "letter": {"static_chain": lambda b, row: P.static_chain(b.chain(row.chain)),
               "text_letters": lambda b, row: P.text_letters(b.chain(row.base_chain), row.max_order, row.gamma),
               "null": lambda b, row: NullLetterInput()},
    "letter_mixer": {"geometric": lambda b, row, mixed: P.geometric_mixer(row, b.loader, mixed),
                     "word_only": lambda b, row, mixed: WordOnlyMixer(len(mixed) - 1)},
    "generic_codec": {"best_of": lambda row: BestOfCodec(row.codecs, [_codec(w, row) for w in row.codecs]),
                      "null": lambda row: NullGenericCodec()},
    "codec": G.CODECS,
    "engine": {"python": E.python_compiler, "numba": E.numba_compiler},
    "fallback_policy": {"smaller": lambda: True, "on_error": lambda: False},
    "integrity": {"none": NullIntegrity, "crc8": Crc8Integrity, "crc16": Crc16Integrity},
    "payload": {"model": lambda asm, mode: ModelPayload(asm.parts(mode)),
                "generic": lambda asm, mode: GenericPayload(asm.generic(mode)),
                "choose": lambda asm, mode: ChoosePayload()},
    "candidates": {"model": lambda row, mode: [mode], "generic": lambda row, mode: [mode],
                   "choose": lambda row, mode: list(row.candidates or [])},
    "table_store": {"eager": lambda s, root: EagerTableStore(table_refs(s, root), _sig_cache(s, root)),
                    "mmap": lambda s, root: MmapTableStore(table_refs(s, root), resolve(root, s.hebzip.table_store_dir),
                                                           _sig_cache(s, root)),
                    "null": lambda s, root: NullTableStore()},
}
"""(drawer family, word) -> builder. Every word the config accepts is a key here (tested).
``fallback_policy``: does compress try the fallbacks (and keep the smallest) even when the primary
mode worked? (``smaller`` yes, ``on_error`` only when it refused) - a table, not an if on the word."""


def table_refs(settings: Settings, root: Path) -> dict[str, TableRef]:
    """Every ``[tables.*]`` as the stores need it: the pinned file and its pinned sha256."""
    return {name: TableRef(resolve(root, row.path), row.sha256) for name, row in settings.tables.items()}


def _sig_cache(settings: Settings, root: Path) -> Path | None:
    rel = settings.hebzip.signature_cache
    return (root / rel) if rel else None


def table_store(settings: Settings, root: Path) -> TableStorePort:
    """The store of the word ``[hebzip].table_store``."""
    store: TableStorePort = REAL["table_store"][settings.hebzip.table_store](settings, root)
    return store


def _codec(word: str, row: ModeRow) -> G.Codec:
    """One codec word ``family[:option...]`` -> (pack, unpack) from the codec registry."""
    family, *options = word.split(":")
    codec: G.Codec = REAL["codec"][family](options, row.ppmd_memory_mb or 0)
    return codec


def engine_kernels(engine: str, root: Path) -> KernelSet:
    """The kernels of the engine word ``engine`` (python | numba), built once per process."""
    return E.kernels(engine, REAL["engine"][engine], root)


def _word(value: str | None, family: str, mode: str) -> str:
    if value is not None:
        return value
    LOG.warning("drawer_defaulted_to_plate", mode=mode, drawer=family)
    return PLATE[family]


class ModeBuild:
    """Builds one model mode's drawers from its rows."""

    def __init__(self, settings: Settings, mode: str, root: Path, store: TableStorePort) -> None:
        self.row: ModeRow = settings.modes[mode]
        self.mode = mode
        language = self.row.language or ""
        self.lang_row = settings.languages[language]
        self.lang = A.language(self.lang_row, language)
        self.loader = A.Loader(store)
        self.k: KernelSet = engine_kernels(settings.hebzip.engine, root)
        self.phantom = settings.hebzip.phantom_bits_max
        self.size = 0
        self.gap: GapClassifierPort = NoGap()
        self.active: list[str] = []
        self._chains: dict[str, LetterChain] = {}

    def parts(self) -> ModeParts:
        """Dictionary, tokenizer, template model, word model, kernels."""
        dictionary = REAL["dictionary"][_word(self.row.dictionary, "dictionary", self.mode)](self)
        self.size = dictionary.size()
        tokenizer = self.tokenizer()
        template_row = self.row.template
        template = REAL["template"][_word(template_row.kind if template_row else None, "template", self.mode)](
            self, template_row)
        self.gap = self.gap_classifier()
        words = REAL["word_model"][_word(self.row.word_model, "word_model", self.mode)](self)
        return ModeParts(dictionary, tokenizer, template, words, self.k, self.phantom, self.gap)

    def tokenizer(self) -> TokenizerPort:
        """The mode's tokenizer drawer (also used by the table builder, ``ops``)."""
        tokenizer: TokenizerPort = REAL["tokenizer"][_word(self.row.tokenizer, "tokenizer", self.mode)](self)
        return tokenizer

    def gap_classifier(self) -> GapClassifierPort:
        """The mode's gap classifier drawer (also used by the table builder, ``ops``)."""
        gap_row = self.row.gap
        gap: GapClassifierPort = REAL["gap"][_word(gap_row.kind if gap_row else None, "gap", self.mode)](
            self, gap_row)
        return gap

    def mixture(self) -> MixtureWordModel:
        """The ``mixture`` word model: base + channels (config order) + mixer."""
        base_row, mixer_row = self.row.base, self.row.mixer
        base = REAL["base"][_word(base_row.kind if base_row else None, "base", self.mode)](self, base_row, self.size)
        channels: list[PredictorPort] = []
        for name in self.row.channels:
            prow = self.row.predictors[name]
            channels.append(REAL["predictor"][_word(prow.kind, "predictor", self.mode)](self, prow))
        self.active = [n for n, p in zip(self.row.channels, channels, strict=True) if p.active]
        mixer = REAL["mixer"][_word(mixer_row.kind if mixer_row else None, "mixer", self.mode)](self, mixer_row)
        coder: WordCoderPort = REAL["word_coder"][_word(self.row.word_coder, "word_coder", self.mode)](self)
        return MixtureWordModel(base, channels, mixer, int(self.row.table_scale_bits or 0), self.k, coder)

    def speller(self) -> WordCoderPort:
        """The word coder ``spelled``: the alphabetical tree, the letter inputs and the letter mixer."""
        row = self.row.speller
        assert row is not None  # checked at start-up (word_coder = 'spelled' needs [speller])
        inputs: list[LetterInputPort] = []
        for name in row.inputs:
            lrow = self.row.letters[name]
            inputs.append(REAL["letter"][_word(lrow.kind, "letter", self.mode)](self, lrow))
        mrow = self.row.letter_mixer
        mixer: LetterMixerPort = REAL["letter_mixer"][_word(mrow.kind if mrow else None, "letter_mixer", self.mode)](
            self, mrow, [True] + [i.active for i in inputs])
        return P.speller(row, self.loader, self.lang, self.loader.words(self.lang_row.word_list), inputs, mixer)

    def chain(self, name: str | None) -> LetterChain:
        """A static letter chain by its ``[letter_chains.<name>]`` (built once, shared by its inputs)."""
        assert name is not None  # checked at start-up
        if name not in self._chains:
            self._chains[name] = P.letter_chain(self.row.letter_chains[name], self.loader, self.lang)
        return self._chains[name]


def _rows(settings: Settings, mode: str) -> dict[str, object]:
    """Every row that shapes the bits of ``mode`` (the mode row minus ``retired``, + its language pack).

    The language's NAME is left out and its rows go in: the signature stays what it was when the
    language was a global row, and a package made with another language's rows is refused (exit 3).
    One plain rule: every row that affects bytes is in the signature, ALWAYS -
    present or absent (an absent row is signed as ``None``). The earlier exceptions (the template
    precision rows left out at format 1's values; ``UNSIGNED_WHEN_ABSENT``, rows added later left out
    when absent) existed only to keep the header byte of packages made before the row existed; the
    public copy has no such packages to protect, so every mode's byte follows from its rows alone."""
    row = dataclasses.asdict(settings.modes[mode])
    row.pop("retired")
    name = row.pop("language")
    lang = dataclasses.asdict(settings.languages[name]) if name is not None else None
    return {"mode": row, "language": lang}


def pinned_signature(settings: Settings, mode: str) -> bytes:
    """The format signature ``mode`` has with this config, from its rows and the PINNED sha256 rows - without
    reading a table. Every store answers ``verify`` with exactly these sha256, so this is the signature of any
    run whose tables pass their check. Used to prove a derived config (``ops/config_subset.py``) signs a mode
    exactly like the full one."""
    shas = {name: settings.tables[name].sha256 for name in mode_tables(settings, mode)}
    return format_signature(_rows(settings, mode), shas)


def _gap_fits(use: Use, settings: Settings) -> None:
    """The ``punctuation`` classifier has one gap class per mark set plus its fixed ones; the mode's
    ``groups`` row must give each class a group. The count comes from the classifier itself
    (``PunctuationGap.classes_for``), not from a second copy of its layout in the config layer."""
    lang = settings.languages[use.mode.language or ""]
    classes = PunctuationGap.classes_for(tuple(lang.gap_marks))
    if len(use.row.groups or []) != classes:
        raise bad("config", f"{use.where}.groups", use.row.groups,
                  f"one group per gap class ({classes} with this language)")


def _adaptive_fits(use: Use, settings: Settings) -> None:
    """The adaptive-table key gives the input number ``AD_INPUT_BITS`` bits: above
    ``AD_INPUTS_MAX`` inputs two would share keys and silently learn into each other's slots, so more is
    refused at start-up. It was measured that the key layout is NOT part of the format (keys are only compared
    for equality; ``AD_INPUT_BITS = 8`` gave 102/102 golden packages): the limit could go up to 256 without
    changing any package - left at 16 as a pending decision, not a format constraint."""
    row = use.row
    count = len(A.cm_spec(row).ad_names())
    if count > AD_INPUTS_MAX:
        raise bad("config", f"{use.where}.adaptive", row.adaptive,
                  f"at most {AD_INPUTS_MAX} adaptive inputs (the word context counts as one), this row gives {count}")


FITS: dict[tuple[str, str], Callable[[Use, Settings], None]] = {("gap", "punctuation"): _gap_fits,
                                                                ("template", "cm"): _adaptive_fits}
"""(family, word) -> a start-up check that needs the drawer's own layout (the declared-dependency rule)."""


def check_fits(settings: Settings) -> None:
    """Every ``FITS`` check of every drawer a mode uses - refuses naming the row."""
    for name, row in settings.modes.items():
        for use in uses(name, row):
            fit = FITS.get((use.family, use.kind))
            if fit is not None:
                fit(use, settings)


class Assembly:
    """One process's view of the config: lazy cached modes, signatures, id lookups."""

    def __init__(self, settings: Settings, root: Path) -> None:
        check_fits(settings)
        self.settings = settings
        self.root = root
        self.store = table_store(settings, root)
        self._parts: dict[str, ModeParts] = {}
        self._sigs: dict[str, bytes] = {}

    def spec(self, mode: str) -> ModeSpec:
        """The facts about ``mode``."""
        row = self.settings.modes.get(mode)
        if row is None:
            raise ConfigError(f"[modes.{mode}]: no such mode")
        return ModeSpec(mode, row.id, row.kind, row.retired)

    def mode_by_id(self, mode_id: int) -> str | None:
        """The mode with this id, or ``None``."""
        return next((n for n, r in self.settings.modes.items() if r.id == mode_id), None)

    def signature(self, mode: str) -> bytes:
        """The mode's format signature (32 bytes) after verifying its tables."""
        if mode not in self._sigs:
            shas = self.store.verify(mode_tables(self.settings, mode))
            self._sigs[mode] = format_signature(_rows(self.settings, mode), shas)
        return self._sigs[mode]

    def parts(self, mode: str) -> ModeParts:
        """The drawers of a model mode (built once, tables verified first)."""
        if self.spec(mode).kind != "model":
            raise ConfigError(f"[modes.{mode}]: not a model mode")
        if mode not in self._parts:
            self.signature(mode)
            self._parts[mode] = ModeBuild(self.settings, mode, self.root, self.store).parts()
            LOG.info("mode_built", mode=mode, engine=self._parts[mode].kernels.engine)
        return self._parts[mode]

    def payload(self, mode: str) -> PayloadCodecPort:
        """The mode as one text <-> payload codec (the only place that looks at the mode's kind)."""
        codec: PayloadCodecPort = REAL["payload"][self.spec(mode).kind](self, mode)
        return codec

    def candidates(self, mode: str) -> list[str]:
        """The modes whose packages compress compares for ``mode``: itself, or a choose mode's list."""
        row = self.settings.modes[mode]
        names: list[str] = REAL["candidates"][row.kind](row, mode)
        return names

    def generic(self, mode: str) -> GenericCodecPort:
        """The codec of a generic mode."""
        row = self.settings.modes[mode]
        if row.kind != "generic":
            raise ConfigError(f"[modes.{mode}]: not a generic mode")
        codec: GenericCodecPort = REAL["generic_codec"][_word(row.generic_codec, "generic_codec", mode)](row)
        return codec

    def fallback_always(self) -> bool:
        """``[hebzip].fallback_policy`` as the flow needs it: try the fallbacks even when the primary worked?"""
        always: bool = REAL["fallback_policy"][self.settings.hebzip.fallback_policy]()
        return always

    def integrity(self, name: str) -> IntegrityPort:
        """The checksum drawer for a config word."""
        drawer: IntegrityPort = REAL["integrity"][name]()
        return drawer

    def integrity_by_code(self, code: int) -> IntegrityPort:
        """The checksum drawer for a header flag value."""
        for cls in REAL["integrity"].values():
            drawer: IntegrityPort = cls()
            if drawer.code == code:
                return drawer
        raise ConfigError(f"integrity code {code}: no such drawer")
