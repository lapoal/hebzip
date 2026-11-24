"""Every drawer's contract (``Protocol``).

Why Protocols with a real implementation AND a null plate (the socket, plate-by-default, conformance-test and
rebuild-from-scratch rules): each
capability of the heart is a drawer that one word in hebzip.toml swaps. A plate answers the
contract in the SAFE direction: a predictor plate is left out of the mixer (contributes nothing),
a template plate codes bytes flat, a tokenizer plate makes no word slots, a word-model plate
refuses. ``@runtime_checkable`` lets the conformance tests ask ``isinstance(plate, Port)``.
The range coder and the payload skeleton are plain code (pure math, no alternative - the change-without-code rule).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

IArr = npt.NDArray[np.int64]


@dataclass(frozen=True)
class KernelSet:
    """The math kernels as the chosen engine built them (plain Python or compiled)."""

    engine: str
    enc_bit: Callable[..., Any]
    dec_bit: Callable[..., Any]
    enc_symbol: Callable[..., Any]
    dec_target: Callable[..., Any]
    dec_symbol: Callable[..., Any]
    dec_start: Callable[..., Any]
    cm_byte: Callable[..., Any]
    flat_byte: Callable[..., Any]
    rehash: Callable[..., Any]
    table_value: Callable[..., Any]
    table_find: Callable[..., Any]
    ctx_seed: Callable[..., Any]
    ctx_roll: Callable[..., Any]
    ctx_finish: Callable[..., Any]
    ctx_word: Callable[..., Any]
    """The template's context hashes: the coder uses them inside ``cm_byte``; the template-prior
    builder calls the python-engine ones on whole numpy arrays - one source for both."""


@dataclass(frozen=True)
class Gap:
    """What stands between the previous word and this one.

    ``cls`` = the gap class (0 = mid-sentence: nothing but spaces; the classifier's other classes
    are its rows) and ``group`` = the class's group (0 = mid, 1 = sentence start, 2 = after a mark in
    the shipped rows). Predictors that react to sentence starts read the group."""

    cls: int = 0
    group: int = 0


MID_GAP = Gap()


@dataclass
class WordHistory:
    """What every word predictor may read about the words so far (owned by the word model).

    ``prev`` = the previous word id (``None`` at the start), ``n`` = words coded so far, ``prev2``
    the word before it and ``gap`` the gap in front of the NEXT word (set before every word) - fields
    nobody reads cost nothing (mode fast)."""

    prev: int | None = None
    prev2: int | None = None
    n: int = 0
    gap: Gap = MID_GAP

    def back(self, k: int) -> int | None:
        """The word ``k`` back (1 = ``prev``, 2 = ``prev2``; the history keeps two - checked at start-up)."""
        return self.prev if k == 1 else self.prev2

    def advance(self, x: int) -> None:
        """Word ``x`` was coded: it becomes the previous word."""
        self.prev2 = self.prev
        self.prev = x
        self.n += 1


@dataclass(frozen=True)
class Look:
    """A predictor's view before the next word: evidence (0 = none) and back-off share to the base.

    ``found``: a CORPUS channel found a list for this context (``corpus_pairs``: the
    previous word is in its table) - the fact the ``trust_selected`` mixer picks a vector by. Channels
    that learn from the text leave it ``False``; no number depends on it."""

    conf: float
    escape: float
    found: bool = False


NO_EVIDENCE = Look(0.0, 0.0)
"""What a predictor says when it has nothing: no evidence, no escape (shared by every predictor)."""


@dataclass(frozen=True)
class Sparse:
    """A sparse part of the coding table: sorted unique ids and their cumulative integer weights."""

    ids: IArr
    cum: IArr = field(repr=False)


@runtime_checkable
class CoderPort(Protocol):
    """What drawers need from the range coder (plain code; ``domain/coder.py``)."""

    encoding: bool
    kernels: KernelSet

    def arrays(self) -> tuple[Any, Any, Any]:
        """(cu, ci, bits) - the shared state the kernels work on."""

    def reserve(self, extra_bits: int) -> None:
        """Make room for ``extra_bits`` more output bits (encoder; no-op when decoding)."""

    def check(self) -> None:
        """Raise ``DamagedStream`` if the decoder ran past its budget or lost its interval."""


@runtime_checkable
class DictionaryPort(Protocol):
    """The word list the tokenizer and the word model share."""

    name: str

    def size(self) -> int:
        """Number of word ids."""

    def lookup(self, word: str) -> int | None:
        """The id of ``word`` or ``None``."""

    def word(self, wid: int) -> str:
        """The word with this id."""


@runtime_checkable
class TokenizerPort(Protocol):
    """Text <-> (template bytes, word ids)."""

    name: str
    slot_end: frozenset[int]

    def split(self, text: str, d: DictionaryPort) -> tuple[bytes, list[int]]:
        """Text -> (template, ids)."""

    def join(self, template: bytes, words: list[str]) -> str:
        """Inverse of ``split``."""

    def slot_start(self, template: bytes | bytearray, after: int, end: int) -> int:
        """Where the slot whose LAST byte is ``template[end]`` begins (searching from ``after``)."""


@runtime_checkable
class GapClassifierPort(Protocol):
    """The gap in front of a word -> its class and group."""

    name: str
    classes: int
    groups: int

    def classify(self, gap: bytes, at_start: bool) -> Gap:
        """``gap`` = the template bytes since the previous slot; ``at_start`` = no word yet."""


@runtime_checkable
class TemplateModelPort(Protocol):
    """Codes the template bytes; learns which word came last (``set_word``)."""

    name: str

    def start(self, n: int) -> None:
        """A new message of ``n`` template bytes."""

    def code_byte(self, coder: CoderPort, byte: int) -> int:
        """Encode ``byte`` / decode one (``byte`` ignored) -> the byte."""

    def set_word(self, wid: int) -> None:
        """The word that just filled a slot."""


@runtime_checkable
class BasePriorPort(Protocol):
    """The dense floor of the word table: every id gets weight >= 1, so every word is codable."""

    name: str
    cum: IArr
    total: int

    def prob(self, x: int) -> float:
        """B(x) = U[x] / total."""


@runtime_checkable
class PredictorPort(Protocol):
    """One mixer channel over word ids. ``active = False`` (the plate) = left out of the mixer."""

    name: str
    trust0: float
    active: bool

    def start(self) -> None:
        """Forget the previous message."""

    def look(self, h: WordHistory) -> Look:
        """Evidence and escape for the next word (cached for ``sparse``/``prob``)."""

    def sparse(self, weight: float, scale: int) -> Sparse | None:
        """Integer weights of the ids this channel knows, at mixer ``weight``; ``None`` = nothing."""

    def prob(self, x: int) -> float:
        """This channel's own probability of ``x`` (without the escape share)."""

    def observe(self, x: int, h: WordHistory) -> None:
        """Word ``x`` was coded (``h`` still describes the moment before it)."""


@runtime_checkable
class MixerPort(Protocol):
    """Turns channel evidence into weights and learns from the coded word."""

    name: str

    def start(self, trust0: list[float]) -> None:
        """A new message with these channels' start trust."""

    def select(self, h: WordHistory, looks: list[Look]) -> None:
        """Before every word: the situation (history + every channel's look) - ``trust_selected``."""

    def weights(self, confs: list[float]) -> list[float]:
        """Weights summing to 1, or all 0.0 when no channel has evidence."""

    def update(self, w: list[float], parts: list[float]) -> None:
        """Learn from each channel's probability of the coded word."""


class CodingTablePort(Protocol):
    """One word's coding table (``domain/mixtable.py``): dense base part ``a * base`` + sparse parts."""

    a: int
    base: BasePriorPort
    parts: list[Sparse]
    total: int

    def code(self, coder: CoderPort, x: int) -> int:
        """Code the word id in one range-coder step."""


@runtime_checkable
class WordCoderPort(Protocol):
    """How one word id goes through the coder, given its table.

    ``direct`` (mode fast; the plate) = one symbol over the whole table; ``spelled`` (mode max) = the
    word letter by letter over the alphabetical tree, mixed with letter models."""

    name: str

    def start(self) -> None:
        """A new message."""

    def code(self, coder: CoderPort, table: CodingTablePort, x: int) -> int:
        """Encode ``x`` / decode one (``x`` ignored) -> the word id."""


@runtime_checkable
class LetterInputPort(Protocol):
    """One letter predictor of the speller. ``active = False`` (the plate) = not mixed.

    ``hist`` = the letter history: the last symbols of the text (oldest first, -1 = nothing yet,
    0 = word end, 1.. = letters) followed by the letters of the word so far. ``reach`` = how many
    symbols back it reads (the speller keeps that much history)."""

    name: str
    active: bool
    reach: int

    def start(self) -> None:
        """Forget the previous message."""

    def logs(self, hist: list[int], codes: list[int]) -> list[float]:
        """log P(next symbol = c) for each child code ``c`` of the tree node."""

    def observe(self, hist: list[int], symbol: int) -> None:
        """``symbol`` followed ``hist`` (called for every letter and the end of each coded word)."""


@runtime_checkable
class LetterMixerPort(Protocol):
    """Mixes the word model's letter view with the letter inputs at one tree node."""

    name: str

    def start(self) -> None:
        """Forget the learnt weights."""

    def select(self, depth: int, masses: list[int], total: int) -> list[float]:
        """The weight set for this node (input 0 = the word model, then the letter inputs)."""

    def probs(self, w: list[float], ins: list[list[float] | None]) -> list[float]:
        """P(child) from the inputs' logs (``None`` = an input that is not mixed)."""

    def learn(self, w: list[float], ins: list[list[float] | None], p: list[float], j: int) -> None:
        """Child ``j`` was coded: move the weights (in place)."""


@runtime_checkable
class WordModelPort(Protocol):
    """Codes one word id per template slot."""

    name: str

    def start(self) -> None:
        """A new message."""

    def code(self, coder: CoderPort, x: int, gap: Gap) -> int:
        """Encode ``x`` / decode one (``x`` ignored) -> the word id; ``gap`` = what stands before it."""


@runtime_checkable
class GenericCodecPort(Protocol):
    """The fallback: a general-purpose compressor frame."""

    name: str

    def encode(self, raw: bytes) -> bytes:
        """Raw bytes -> payload (codec byte + varint length + packed)."""

    def decode(self, payload: bytes) -> bytes:
        """Payload -> raw bytes."""

    def codec_names(self) -> list[str]:
        """The codecs this drawer can pick from (their list is part of the format signature)."""


@runtime_checkable
class HolesPort(Protocol):
    """Words sent as a page number and completed by a decoder model - contract only."""

    name: str

    def choose(self, slots: list[int], context: str) -> dict[int, str]:
        """Which slots to leave as holes -> {slot: word}; ``{}`` = no holes."""


@runtime_checkable
class IntegrityPort(Protocol):
    """Optional checksum of the payload. ``code`` is the header flag value."""

    name: str
    code: int
    size: int

    def compute(self, payload: bytes) -> bytes:
        """The ``size`` checksum bytes of ``payload`` (``b""`` for the null plate)."""


@runtime_checkable
class PayloadCodecPort(Protocol):
    """A whole mode as the flows see it: text -> payload and back, whatever it is built from.

    Chosen ONCE by the selector from the mode's ``kind`` (``model`` / ``generic``), so no flow asks
    which kind of mode it serves (the choose-once-at-the-root rule). ``domain/payload.py``."""

    name: str

    def encode(self, text: str) -> bytes:
        """Text -> payload (without the header)."""

    def decode(self, payload: bytes, limit: int) -> bytes:
        """Payload -> the original UTF-8 bytes; at most ``limit`` bytes may be declared."""


@dataclass(frozen=True)
class ModeParts:
    """What a model mode is built from: its drawers, its kernels and the decoder's damage budget."""

    dictionary: DictionaryPort
    tokenizer: TokenizerPort
    template: TemplateModelPort
    words: WordModelPort
    kernels: KernelSet
    phantom_max: int
    gap: GapClassifierPort
