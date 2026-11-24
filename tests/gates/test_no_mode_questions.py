"""Gate (the one-selector rule and the choose-once-at-the-root rule and the change-without-code rule): only
the selector turns a word of the config into behaviour.

Why: ``services/compress.py`` and ``services/decompress.py`` each branched on
``spec.kind == "generic"`` - two flows that knew which product they were serving.
Why the second test: a review still found drawer words decided by if-chains
outside the selector - the table list (``config/tables.py``: ``row.word_coder == "spelled"``), the
engine (``services/engines.py``: ``engine == "numba"``) and the generic codecs
(``integrations/generic.py``: ``family == "deflate"`` ...). Each new drawer then meant editing an
if-chain somewhere. Now every word is a key of a table (a registry) that the one selector
(``services/select.py``) or a row table resolves; nothing else compares against a word.
AST based: a comparison against a ``.kind`` attribute, a mode name, or any word the config
vocabulary defines (drawer kinds, engines, codecs, policies, recipes ...), or ``startswith`` /
``endswith`` with such a word, fails - in every layer except the selector. So does a
``match``/``case`` on a word and a comparison against a module constant that holds a word.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from hebzip.config import schema as S
from tests.support import CONFIG, ROOT

PACKAGE = ROOT / "hebzip"
LAYERS = ("domain", "services", "interface")
SELECTOR = PACKAGE / "services" / "select.py"
KIND_WORDS = {"model", "generic"}


def _mode_words() -> set[str]:
    with CONFIG.open("rb") as handle:
        return set(tomllib.load(handle)["modes"]) | KIND_WORDS


def drawer_words() -> set[str]:
    """Every word the config may use to choose a behaviour (from the schema's word tables)."""
    words: set[str] = set(S.KINDS) | set(S.ENGINE_WORDS) | set(S.INTEGRITY_WORDS) | set(S.HOLES_WORDS)
    words |= set(S.FALLBACK_POLICIES) | set(S.PLATE.values()) | set(S.CHAIN_CONTEXTS) | set(S.RECIPE_ROWS)
    words |= set(S.CODEC_FORMS) | set(S.CODEC_OPTIONS)
    words |= set(S.STORE_WORDS)  # eager | mmap | null
    for table in (S.MODE_WORDS, S.KIND_ROWS):
        for family in table.values():
            words |= set(family)
    return words | _mode_words()


def _constants(side: ast.expr) -> list[str]:
    if isinstance(side, ast.Constant) and isinstance(side.value, str):
        return [side.value]
    if isinstance(side, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in side.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _named_words(tree: ast.Module, words: set[str]) -> set[str]:
    """Module constants that hold a word (``NUMBA_WORD = 'numba'``, ``WORDS = ('numba', ...)``)."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(c in words for c in _constants(node.value)):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return out


def _asks(side: ast.expr, words: set[str], named: set[str]) -> bool:
    if isinstance(side, ast.Attribute) and side.attr == "kind":
        return True
    if isinstance(side, ast.Name) and side.id in named:
        return True
    return any(c in words for c in _constants(side))


def _asked(node: ast.AST, words: set[str], named: set[str]) -> bool:
    """Does this node decide on a word: a comparison, a prefix test or a ``case`` pattern?"""
    if isinstance(node, ast.Compare):
        return any(_asks(s, words, named) for s in [node.left, *node.comparators])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith",
                                                                                                 "endswith"):
        return any(c in words for a in node.args for c in _constants(a))
    if isinstance(node, ast.MatchValue):
        return _asks(node.value, words, named)
    return False


def _where(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix() if path.is_relative_to(PACKAGE) else path.name


def questions(files: list[Path], words: set[str]) -> list[str]:
    """Every comparison, prefix test or ``case`` that asks which mode, kind or drawer word (empty = none)."""
    out: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        named = _named_words(tree, words)
        out += [f"{_where(path)}:{getattr(node, 'lineno', 0)} {ast.unparse(node)}"
                for node in ast.walk(tree) if _asked(node, words, named)]
    return out


def _files(layers: tuple[str, ...] | None = None) -> list[Path]:
    roots = [PACKAGE / layer for layer in layers] if layers else [PACKAGE]
    found = sorted(p for root in roots for p in root.rglob("*.py") if "__pycache__" not in p.parts and p != SELECTOR)
    assert len(found) > 20
    return found


def test_no_flow_asks_which_mode_it_serves() -> None:
    found = questions(_files(LAYERS), _mode_words())
    assert not found, found


def test_no_module_branches_on_a_drawer_word() -> None:
    assert {"spelled", "numba", "deflate", "punctuation", "template_prior"} <= drawer_words()
    found = questions(_files(), drawer_words())
    assert not found, found


EVASIONS = {
    "match": "def q(engine: str) -> bool:\n    match engine:\n        case 'numba':\n            return True\n"
             "    return False\n",
    "named": "NUMBA_WORD = 'numba'\n\n\ndef q(engine: str) -> bool:\n    return engine == NUMBA_WORD\n",
    "named_in": "WORDS = ('numba', 'python')\n\n\ndef q(engine: str) -> bool:\n    return engine in WORDS\n",
}


def test_the_gate_sees_match_and_named_words(tmp_path: Path) -> None:
    """A ``match``/``case 'numba'`` or a comparison against a module constant that
    holds the word (``NUMBA_WORD = 'numba'``) decided on a word and passed the gate unseen."""
    for name, code in EVASIONS.items():
        path = tmp_path / f"{name}.py"
        path.write_text(code, encoding="utf-8")
        assert questions([path], drawer_words()), name
