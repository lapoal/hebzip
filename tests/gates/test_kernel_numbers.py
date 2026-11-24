"""Gate (the change-without-code rule "a detail is a row"): no precision number comes back into the
template kernel.

Why: the template precision moved (12-bit probabilities, 16-bit fixed point)
into the rows ``prob_bits`` / ``fix_bits``. Nothing guarded that: putting ``2047`` back in place of
``ip[P_SPAN]`` in ``mix`` kept every test green - encoder and decoder run the same wrong code, so the
round trips stay exact, the golden runs use 12/16 where the literal is right, and a package made at
another precision still differs from the default one. The change-without-code rule needs a check that fails on such
code.

AST based, over the template model's code (kernel, its tables, its drawer, its corpus recipe) and the
coder kernel it calls: every integer written there must be structure, not a detail -
``0``/``1``/``2``, the bits of a byte (``7``, ``8``, ``255``), or the value of one of the named module
constants in ``NAMED`` (each with the reason it is not a row). A tuple index (``bigger[3]``) is
structure too. Anything else (``12``, ``16``, ``2047``, ``4096``, ``65536`` ...) fails, naming the place.
``domain/kernels_layout.py`` is not scanned: it is the one table of format slots and widths.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support import ROOT

PACKAGE = ROOT / "hebzip"
SCANNED = ("domain/kernels/cm.py", "domain/kernels/rc.py", "domain/kernels/arith.py", "domain/template_tables.py",
           "domain/template_cm.py", "domain/template_fit.py", "ops/template_prior.py")
STRUCTURE = {0, 1, 2, 7, 8, 255}
"""0/1/2 and the bits of a byte (MSB index 7, 8 bits, mask 255)."""
NAMED = {
    "MARK": "adaptive key layout bit - part of the format",
    "START_SLOTS": "notebook start size - memory only, no bit depends on it",
    "BYTE_BITS_RESERVE": "coder buffer reserve per template byte - memory only",
}


def _allowed_nodes(tree: ast.Module) -> set[int]:
    """ids of the integer constants that are structure by position (named constants, tuple indexes)."""
    ok: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in NAMED for t in targets) and node.value is not None:
                ok |= {id(n) for n in ast.walk(node.value)}
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
            ok.add(id(sub.slice))
    return ok


def numbers(path: Path, label: str) -> list[str]:
    """Every integer literal in ``path`` that is not structure (empty = clean)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ok = _allowed_nodes(tree)
    return [f"{label}:{n.lineno} {n.value}" for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and type(n.value) is int and n.value not in STRUCTURE and id(n) not in ok]


def test_the_template_kernel_holds_no_precision_number() -> None:
    files = [PACKAGE / rel for rel in SCANNED]
    assert all(p.is_file() for p in files), [str(p) for p in files if not p.is_file()]
    found = [hit for path, rel in zip(files, SCANNED, strict=True) for hit in numbers(path, rel)]
    assert not found, found


def test_the_gate_catches_a_precision_number_put_back(tmp_path: Path) -> None:
    """The plants: a literal in place of a precision slot, and a precision constant."""
    kernel = (PACKAGE / "domain" / "kernels" / "cm.py").read_text(encoding="utf-8")
    assert "return sqt[d + ip[P_SPAN]]" in kernel
    for name, code in {"literal": kernel.replace("return sqt[d + ip[P_SPAN]]", "return sqt[d + 2047]"),
                       "constant": kernel + "\nPROB_BITS = 12\n"}.items():
        path = tmp_path / f"{name}.py"
        path.write_text(code, encoding="utf-8")
        assert numbers(path, name), name
