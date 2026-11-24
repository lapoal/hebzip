"""Code rules as a gate (the code rules): files <= 400 lines, functions <= 50, no
``print``, no bare/``Exception`` except, no ``sys.path`` anywhere (the research prototype folders are gone
from the path for good), the time only in ``core/clock.py``; then ``mypy --strict hebzip`` and
``ruff check`` must exit 0.
AST based, so a word in a string or a docstring cannot trigger or hide a violation."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import ROOT

MAX_FILE, MAX_FUNC = 400, 50
SCANNED = [ROOT / "hebzip", ROOT / "tests"]
CLOCK_FILE = ROOT / "hebzip" / "core" / "clock.py"
TIME_CALLS = {("time", "time"), ("datetime", "now"), ("datetime", "utcnow"), ("datetime", "today")}


def _files() -> list[Path]:
    found = [p for base in SCANNED for p in base.rglob("*.py")
             if not {"_work", "__pycache__"} & set(p.relative_to(ROOT).parts)]
    assert len(found) > 30
    return sorted(found)


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted(node: ast.expr) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ("Exception", "BaseException") for n in names)


def _length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return (node.end_lineno or 0) - node.lineno + 1


def _violations(path: Path) -> list[str]:
    tree, out = _parsed(path), []
    lines = len(path.read_text(encoding="utf-8").splitlines())
    if lines > MAX_FILE:
        out.append(f"{path.name}: {lines} lines > {MAX_FILE}")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _length(node) > MAX_FUNC:
            out.append(f"{path.name}:{node.lineno} function {node.name} longer than {MAX_FUNC}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            out.append(f"{path.name}:{node.lineno} print()")
        if isinstance(node, ast.ExceptHandler) and _broad(node):
            out.append(f"{path.name}:{node.lineno} broad except")
        if isinstance(node, ast.Attribute):
            out += _attribute_violations(path, node)
    return out


def _attribute_violations(path: Path, node: ast.Attribute) -> list[str]:
    dotted = _dotted(node)
    if dotted[-2:] == ("sys", "path"):
        return [f"{path.name}:{node.lineno} sys.path (forbidden: nothing outside the project is imported)"]
    if dotted[-2:] in TIME_CALLS and path != CLOCK_FILE:
        return [f"{path.name}:{node.lineno} reads the clock outside core/clock.py"]
    return []


def test_code_rules_hold_everywhere() -> None:
    problems = [v for path in _files() for v in _violations(path)]
    assert not problems, problems


def _tool(*args: str) -> subprocess.CompletedProcess[str]:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    python = str(venv) if venv.exists() else sys.executable
    return subprocess.run([python, "-m", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          timeout=900, check=False)


def test_mypy_strict_is_clean() -> None:
    done = _tool("mypy", "--strict", "hebzip")
    if "No module named mypy" in done.stderr:
        pytest.fail("mypy is not installed (python -m venv .venv; .venv/Scripts/pip install mypy)")
    assert done.returncode == 0, done.stdout[-3000:]


def test_mypy_strict_is_clean_on_the_tests() -> None:
    """The code rules (<= 400 / <= 50, mypy strict) - "mypy strict ... applies to EVERYTHING, tests
    included"; the gate
    checked only ``hebzip`` and two test files had drifted (an untyped dict, ``.lineno`` on ``ast.AST``)."""
    done = _tool("mypy", "--strict", "--explicit-package-bases", "tests")
    assert done.returncode == 0, done.stdout[-3000:]


def test_ruff_is_clean() -> None:
    done = _tool("ruff", "check", "hebzip", "tests")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-500:]
