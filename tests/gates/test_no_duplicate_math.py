"""Gate (design decision, "one source"): the same math is not written twice in ``hebzip``.

Why: a review found the cumulative sum written twice (``words/base.py`` and ``words/notebook.py``) and the
coder's register constants defined in three files. Two copies drift apart silently - one gets a fix,
the other keeps the bug. Two witnesses, AST based (a docstring or a comment cannot hide or trigger):
1. function bodies: the same body (docstring dropped, parameter names replaced by their position)
   in two functions = one formula written twice. Bodies smaller than ``MIN_NODES`` AST nodes are
   skipped: ``return 0.0`` in two plates is not a formula.
2. module constants: the same NAME bound to the same VALUE in two modules (``FULL = (1 << 64) - 1``
   here, ``FULL = U(0xFFFF...)`` there) = one fact in two ledgers (the one-ledger-per-fact rule).
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from pathlib import Path

from tests.support import ROOT

PACKAGE = ROOT / "hebzip"
MIN_NODES = 12
"""Smallest body that counts as a formula (``return 0.0`` has 3 nodes, the cumulative sum 22)."""
WRAPPERS = ("U", "uint64", "int64")
"""Calls that only change an integer's type: ``U(1 << 62)`` is the value ``1 << 62``."""
OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.LShift: operator.lshift,
    ast.RShift: operator.rshift, ast.BitOr: operator.or_, ast.BitAnd: operator.and_, ast.Pow: operator.pow}


def _files() -> list[Path]:
    found = sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(found) > 30
    return found


def _body_key(fn: ast.FunctionDef) -> str | None:
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not body:
        return None
    params = {a.arg: f"_p{i}" for i, a in enumerate(fn.args.args + fn.args.kwonlyargs)}
    module = ast.Module(body=[ast.parse(ast.unparse(s)).body[0] for s in body], type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id in params:
            node.id = params[node.id]
    if sum(1 for _ in ast.walk(module)) < MIN_NODES:
        return None
    return ast.dump(module)


def _value(node: ast.expr) -> object:
    """An integer constant expression -> its value; anything else -> its AST dump."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _value(node.left), _value(node.right)
        if isinstance(left, int) and isinstance(right, int):
            return OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and len(node.args) == 1 and ast.unparse(node.func).split(".")[-1] in WRAPPERS:
        return _value(node.args[0])
    return ast.dump(node)


def _constants(tree: ast.Module) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        else:
            continue
        if target.isupper() and not isinstance(value, ast.Subscript | ast.Attribute | ast.Name):
            out[target] = _value(value)
    return out


def duplicates(files: list[Path]) -> list[str]:
    """Every formula / constant written in more than one place (empty = one source everywhere)."""
    bodies: dict[str, str] = {}
    consts: dict[tuple[str, str], str] = {}
    out: list[str] = []
    for path in files:
        rel = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            key = _body_key(fn)
            if key is not None and key in bodies:
                out.append(f"{rel}:{fn.lineno} {fn.name} repeats {bodies[key]}")
            elif key is not None:
                bodies[key] = f"{rel}:{fn.lineno} {fn.name}"
        for name, value in _constants(tree).items():
            ident = (name, repr(value))
            if ident in consts:
                out.append(f"{rel}: constant {name} repeats {consts[ident]}")
            else:
                consts[ident] = rel
    return out


def test_every_formula_and_constant_has_one_source() -> None:
    found = duplicates(_files())
    assert not found, found
