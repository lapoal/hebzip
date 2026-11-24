"""Layer boundaries as a gate (the layer table import table).

Why a gate (stage-3 review): the import table of was only a promise - nothing failed
when ``config`` imported ``domain`` or ``domain`` touched the disk. AST based: every ``import`` /
``from ... import`` inside ``hebzip/<layer>/`` must stay within the layers that layer may use, and
``domain`` (pure rules, zero I/O) may not import an I/O module nor call ``open``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support import ROOT

PACKAGE = ROOT / "hebzip"
ALLOWED: dict[str, set[str]] = {
    "core": {"core"},
    "config": {"core", "config"},
    "domain": {"core", "domain"},
    "store": {"core", "store"},
    "integrations": {"core", "domain", "integrations"},
    "services": {"core", "config", "domain", "store", "integrations", "services"},
    "interface": {"core", "config", "domain", "services", "interface"},
    "ops": {"core", "config", "domain", "store", "services", "ops"},
}
ROOT_FILES = {"__init__.py", "__main__.py"}
"""The only modules allowed directly under ``hebzip/`` (outside every layer)."""
DOMAIN_IO = {"os", "io", "pathlib", "sys", "subprocess", "socket", "shutil", "tempfile", "time", "datetime",
             "urllib", "http", "logging", "importlib", "glob", "numba", "zipfile", "pickle"}


DOMAIN_CALLS = {"load", "save", "savez", "fromfile", "tofile", "loadtxt", "read_text", "read_bytes", "write_text",
                "write_bytes"}
"""numpy / pathlib calls that touch the disk - forbidden in ``domain`` even without an I/O import."""


def _imports(tree: ast.Module, package: list[str]) -> list[tuple[int, str]]:
    """Every imported module as an absolute name. ``package`` = the file's package (``hebzip.domain...``):
    a relative import (``from ..services import x``) is resolved against it (it used to be skipped)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            base = package[:len(package) - node.level + 1]
            if node.module:
                found.append((node.lineno, ".".join(base + [node.module])))
            else:
                found += [(node.lineno, ".".join(base + [alias.name])) for alias in node.names]
    return found


def _layer_problems(layer: str, path: Path, tree: ast.Module, package: list[str]) -> list[str]:
    out = []
    for line, module in _imports(tree, package):
        parts = module.split(".")
        if parts[0] == "hebzip" and len(parts) > 1 and parts[1] not in ALLOWED[layer]:
            out.append(f"{layer}/{path.name}:{line} imports {module}")
        if layer == "domain" and parts[0] in DOMAIN_IO:
            out.append(f"domain/{path.name}:{line} imports I/O module {module}")
    if layer == "domain":
        names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "open"]
        out += [f"domain/{path.name}:{n.lineno} open" for n in names]
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in DOMAIN_CALLS]
        out += [f"domain/{path.name}:{n.lineno} {n.attr}" for n in calls]
    return out


def problems(package: Path = PACKAGE) -> list[str]:
    """Every boundary violation under ``package`` (empty = clean)."""
    out: list[str] = []
    for layer in ALLOWED:
        for path in sorted((package / layer).rglob("*.py")):
            if "__pycache__" not in path.parts:
                inner = list(path.relative_to(package).parent.parts)
                out += _layer_problems(layer, path, ast.parse(path.read_text(encoding="utf-8")),
                                       [package.name, *inner])
    return out


def strays(package: Path = PACKAGE) -> list[str]:
    """Folders and modules under ``package`` that belong to no layer (an unknown folder would never be scanned)."""
    out = [p.name for p in package.iterdir() if p.is_dir() and p.name != "__pycache__" and p.name not in ALLOWED]
    out += [p.name for p in package.glob("*.py") if p.name not in ROOT_FILES]
    return out


def test_every_layer_imports_only_what_it_may() -> None:
    assert (PACKAGE / "domain" / "header.py").is_file()
    assert not problems(), problems()


def test_every_folder_is_a_known_layer() -> None:
    assert not strays(), strays()


def test_a_relative_import_across_layers_is_caught(tmp_path: Path) -> None:
    """``from ..services import select`` inside ``domain`` passed the gate - only
    absolute imports were checked. The gate must see through the dots."""
    domain = tmp_path / "hebzip" / "domain" / "sub"
    domain.mkdir(parents=True)
    (domain / "x.py").write_text("from ...services import select\nfrom .. import header\n", encoding="utf-8")
    (tmp_path / "hebzip" / "domain" / "y.py").write_text("from .. import services\n", encoding="utf-8")
    found = problems(tmp_path / "hebzip")
    assert any("x.py" in p and "hebzip.services" in p for p in found), found
    assert any("y.py" in p and "hebzip.services" in p for p in found), found
    assert not any("hebzip.domain.header" in p for p in found), found
