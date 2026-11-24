"""Pure table builders for a language pack: corpus counts -> the arrays the drawers read (zero I/O).

Why this exists (the change-without-code rule: "another country"): a new language must be rows + tables, and the
tables must be buildable without the research prototype. These functions build the word-level tables in
exactly the layout the builders in ``services/assemble*.py`` check, and they reuse the domain's own
pieces (tokenizer, gap classifier through ``stream.gap_before``, ``keys_from_hist``) so a table can
never disagree with the code that reads it. Ids are frequency ranks (the Zipf part of the base
assumes it); ties by word, so the same corpus always gives the same tables.
The corpus is a list of documents (an article, a line) - word pairs, gap starts and the template
prior never reach across a document border, like the research prototype's per-article recipes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from hebzip.domain.contracts import Gap, GapClassifierPort, TokenizerPort
from hebzip.domain.spell.chains import FAMILY, WITHIN, keys_from_hist
from hebzip.domain.spell.trie import END
from hebzip.domain.stream import gap_before

IArr = npt.NDArray[np.int64]
Arrays = dict[str, npt.NDArray[np.generic]]
Doc = tuple[bytes, list[int]]
"""One tokenized document: its template and the word id of every slot."""


class CollectingDictionary:
    """A ``DictionaryPort`` that knows every word it is asked about (ids in order of first sight)."""

    name = "collecting"

    def __init__(self) -> None:
        self.words: list[str] = []
        self.index: dict[str, int] = {}

    def size(self) -> int:
        """Words seen so far."""
        return len(self.words)

    def lookup(self, word: str) -> int | None:
        """Every word gets an id."""
        if word not in self.index:
            self.index[word] = len(self.words)
            self.words.append(word)
        return self.index[word]

    def word(self, wid: int) -> str:
        """The word with this id."""
        return self.words[wid]


def ranked_words(texts: list[str], tokenizer: TokenizerPort) -> list[str]:
    """Every dictionary form in ``texts``, most frequent first (ties by the word itself)."""
    seen = CollectingDictionary()
    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(tokenizer.split(text, seen)[1])
    return sorted(seen.words, key=lambda w: (-counts[seen.index[w]], w))


def ints(values: list[int]) -> IArr:
    """A list of ints as an int64 array."""
    return np.array(values, dtype=np.int64)


def pair_arrays(docs: list[Doc], size: int) -> Arrays:
    """``uni`` (count per id) + the follower lists of ``corpus_pairs`` (ctx, ctx_n, off, nxt_d, cnt)."""
    follow: dict[int, Counter[int]] = {}
    uni = np.zeros(size, dtype=np.int64)
    for _template, ids in docs:
        for a, b in zip(ids, ids[1:], strict=False):
            follow.setdefault(a, Counter())[b] += 1
        uni += np.bincount(ints(ids), minlength=size).astype(np.int64)
    ctx = sorted(follow)
    off, nxt_d, cnt, ctx_n = [0], [], [], []
    for c in ctx:
        prev = 0
        for x, k in sorted(follow[c].items()):
            nxt_d.append(x - prev)
            cnt.append(k)
            prev = x
        off.append(len(cnt))
        ctx_n.append(sum(follow[c].values()))
    return {"uni": uni, "ctx": ints(ctx), "ctx_n": ints(ctx_n), "off": ints(off), "nxt_d": ints(nxt_d),
            "cnt": ints(cnt)}


def slot_gaps(template: bytes, tokenizer: TokenizerPort, gap: GapClassifierPort) -> list[Gap]:
    """The gap in front of every slot of one document, classified exactly as the coder does."""
    out: list[Gap] = []
    after = 0
    for i, byte in enumerate(template):
        if byte in tokenizer.slot_end:
            out.append(gap_before(tokenizer, gap, template, after, i, not out))
            after = i + 1
    return out


def gap_start_arrays(docs: list[Doc], tokenizer: TokenizerPort, gap: GapClassifierPort,
                     groups: Collection[int], keep_top: int) -> Arrays:
    """``ids{g}`` / ``cnt{g}`` / ``N{g}`` per gap class g: the words after that gap (classes of the
    predictor's ``groups`` only), the ``keep_top`` most frequent kept (ties: smaller id), N = all."""
    per_class: list[Counter[int]] = [Counter() for _ in range(gap.classes)]
    for template, ids in docs:
        for found, x in zip(slot_gaps(template, tokenizer, gap), ids, strict=True):
            if found.group in groups:
                per_class[found.cls][x] += 1
    out: Arrays = {}
    for g, counts in enumerate(per_class):
        kept = sorted(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:keep_top])
        out[f"ids{g}"] = ints([x for x, _ in kept])
        out[f"cnt{g}"] = ints([k for _, k in kept])
        out[f"N{g}"] = ints([sum(counts.values())])
    return out


def letter_stream(words: list[str], alphabet: str) -> list[int]:
    """The speller's symbols for a word sequence: letter codes 1.., then ``END`` after every word."""
    code = {ch: i + 1 for i, ch in enumerate(alphabet)}
    out: list[int] = []
    for w in words:
        out += [code[ch] for ch in w]
        out.append(END)
    return out


def chain_arrays(streams: list[list[int]], symbols: int, context: str, max_order: int, symbol_bits: int) -> Arrays:
    """``n0`` + ``<f>{k}_ck`` / ``<f>{k}_cn`` (k = 1..max_order) of one context family over the documents'
    symbol streams. The history starts empty in every document (like the research prototype's per-article
    padding - a coded text also starts with an empty history)."""
    within = WITHIN[context]
    base = symbols + 1
    counts: list[Counter[int]] = [Counter() for _ in range(max_order)]
    n0 = np.zeros(symbols, dtype=np.int64)
    for stream in streams:
        hist = [-1] * max_order
        for sym in stream:
            for k, key in enumerate(keys_from_hist(hist, max_order, within, base)):
                counts[k][key << symbol_bits | sym] += 1
            hist = hist[1:] + [sym]
        n0 += np.bincount(ints(stream), minlength=symbols).astype(np.int64)
    family = FAMILY[context]
    out: Arrays = {"n0": n0}
    for k, found in enumerate(counts, start=1):
        pairs = sorted(found.items())
        out[f"{family}{k}_ck"] = ints([a for a, _ in pairs])
        out[f"{family}{k}_cn"] = ints([b for _, b in pairs])
    return out


def trie_arrays(words: list[str], alphabet: str, columns: int) -> Arrays:
    """``perm`` (word id per alphabetical rank) + ``cols`` (letter code per depth and rank, 0 = ended) for the
    first ``columns`` letters - deeper letters the speller finds by search (same result, slower)."""
    code = {ch: i + 1 for i, ch in enumerate(alphabet)}
    perm = sorted(range(len(words)), key=lambda x: words[x])
    cols = np.zeros((columns, len(words)), dtype=np.uint8)
    for rank, wid in enumerate(perm):
        for k, ch in enumerate(words[wid][:columns]):
            cols[k, rank] = code[ch]
    return {"perm": ints(perm), "cols": cols}
