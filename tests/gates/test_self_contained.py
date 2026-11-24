"""Gate: the tool is self-contained - it reads and imports nothing from outside this repository.

Why (design decision, zero exceptions): the heart was rebuilt without the earlier research code, and a
public copy must run from any folder. Three independent witnesses:
1. AST: no ``sys.path`` anywhere and no ``importlib`` call outside ``hebzip.*`` / numba - in ``hebzip/``
   and ``tests/`` (a path trick is how outside code would come back in).
2. The config: every table is a relative path inside ``tables/`` (no absolute path, no ``..``).
3. A real run: compress + decompress in a child process, then no loaded module comes from outside the
   repository's own code, the interpreter's environment (venv) and the standard library.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

from tests.support import CONFIG, ROOT, TEXTS, work_dir

ALLOWED_DYNAMIC = ("hebzip.", "numba")
RUN_TEXT = "own_story"


def _sources() -> list[Path]:
    files = [p for base in (ROOT / "hebzip", ROOT / "tests") for p in base.rglob("*.py")
             if not {"_work", "__pycache__"} & set(p.relative_to(ROOT).parts)]
    assert len(files) > 40
    return sorted(files)


def _problems(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) \
                and node.value.id == "sys":
            out.append(f"{path.name}:{node.lineno} sys.path")
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module" and node.args:
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and str(arg.value).startswith(ALLOWED_DYNAMIC)):
                out.append(f"{path.name}:{node.lineno} import_module of something else")
    return out


def test_no_source_reaches_outside_by_a_path_trick() -> None:
    problems = [p for path in _sources() for p in _problems(path)]
    assert not problems, problems


def test_the_gate_sees_a_path_trick(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("import sys\nsys.path.insert(0, 'x')\n", encoding="utf-8")
    assert _problems(probe) == ["probe.py:2 sys.path"]


def test_config_reads_only_project_files() -> None:
    for name, row in tomllib.loads(CONFIG.read_text(encoding="utf-8"))["tables"].items():
        path = Path(row["path"])
        assert not path.is_absolute() and path.parts[0] == "tables" and ".." not in path.parts, name


CHILD = """
import sys, pathlib
from hebzip.interface.cli import main
src, dst, back = sys.argv[1], sys.argv[2], sys.argv[3]
for mode in ("fast", "max"):
    assert main(["compress", "--mode", mode, "--no-fallback", "--force", src, dst]) == 0
    assert main(["decompress", "--force", dst, back]) == 0
root, venv = pathlib.Path(sys.argv[4]), pathlib.Path(sys.prefix).resolve()
base = pathlib.Path(sys.base_prefix).resolve()
outside = []
for name, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if not f:
        continue
    p = pathlib.Path(f).resolve()
    if not (p.is_relative_to(root / "hebzip") or p.is_relative_to(venv) or p.is_relative_to(base)):
        outside.append(str(p))
assert not outside, outside
"""


def test_a_real_run_loads_nothing_from_outside() -> None:
    work = work_dir("self_contained_run")
    done = subprocess.run([sys.executable, "-c", CHILD, str(TEXTS / f"{RUN_TEXT}.txt"), str(work / "x.hz"),
                           str(work / "x.out"), str(ROOT)], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-1500:]
