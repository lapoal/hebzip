"""Shared helpers for the tests: run the real CLI in a child process, golden files, config variants.

Why child processes (the real-path rule and the one-process-per-test-file rule): the user runs ``python -m
hebzip``; a test that calls a
function inside its own interpreter would pass on state the real run does not have, and a decoder
must see only package bytes. Scratch files go to ``tests/_work`` (git-ignored) and are removed when
the test ends (``clean_work``).
Config variants are made by replacing exact text of the real config (there is no TOML writer):
every replacement must match exactly once, so a moved row fails loudly instead of testing nothing.
"""

from __future__ import annotations

import copy
import inspect
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden"
TEXTS = GOLDEN / "texts"
CONFIG = ROOT / "config" / "hebzip.toml"
WORK = ROOT / "tests" / "_work"
CLI_TIMEOUT_S = 900
SIGNATURE_BYTE = {"fast": 0x94, "max": 0x81, "max_plus": 0x5C, "generic": 0x66, "latin_fast": 0x32,
                  "latin_max": 0xE9, "en_fast": 0x9E, "en_max": 0x9D}
"""The header's signature byte of every mode with the real config (the template precision rows are always in
the signature). Written out per mode so a changed row, table or signature rule is seen as a changed byte, also
by tests that do not read the golden packages."""
ENGINES = ("python", "numba")
TEXT_SETS = {"he": TEXTS, "en": GOLDEN / "texts_en", "latin_toy": ROOT / "langpacks" / "latin_toy" / "samples"}
"""The golden texts of each language (the Latin toy pack's samples live with the pack)."""
GOLDEN_MODES = {"fast": "he", "max": "he", "max_plus": "he", "generic": "he", "en_fast": "en", "en_max": "en",
                "latin_fast": "latin_toy", "latin_max": "latin_toy"}
"""Every mode of the real config -> the text set its golden packages were made from."""
HEBREW_TEXTS = 13
"""8 texts written for this project (``own_*``) + 5 Wikipedia article openings (``wiki_*``)."""
SAMPLE = ("own_story", "wiki_cat", "own_message")
"""Three texts (short story, encyclopedic, short message) for the slower per-drawer runs."""


def text_paths() -> list[Path]:
    """The Hebrew golden texts, in a stable order."""
    return sorted(TEXTS.glob("*.txt"))


def mode_texts(mode: str) -> list[Path]:
    """The golden texts of ``mode`` (its language's set), in a stable order."""
    return sorted(TEXT_SETS[GOLDEN_MODES[mode]].glob("*.txt"))


def golden_package(mode: str, stem: str) -> bytes:
    """The golden package of text ``stem`` in ``mode`` - the whole file the CLI wrote (header + payload)."""
    return (GOLDEN / mode / f"{stem}.hz").read_bytes()


def sample_paths() -> list[Path]:
    """The ``SAMPLE`` texts."""
    return [TEXTS / f"{name}.txt" for name in SAMPLE]


def run_cli(*args: str | Path, timeout: int = CLI_TIMEOUT_S) -> subprocess.CompletedProcess[bytes]:
    """``python -m hebzip <args>`` in a fresh interpreter, cwd = project root."""
    cmd = [sys.executable, "-m", "hebzip", *map(str, args)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=timeout, check=False)


_MADE: list[Path] = []
"""Scratch folders made during the running test - removed by ``clean_work`` when it ends."""


def work_dir(name: str) -> Path:
    """A fresh empty scratch folder ``tests/_work/<name>``, removed again when the test ends.

    Why removed at the END (the machine-and-material rule): the folders used to be wiped only at the
    START of the next run, so decoded copies and packages of private texts stayed on disk
    between runs. ``tests/conftest.py`` calls ``clean_work`` after every test - also a failed one.
    """
    path = WORK / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    _MADE.append(path)
    return path


def clean_work() -> list[Path]:
    """Remove every scratch folder made since the last call; returns the ones that could not be removed."""
    left: list[Path] = []
    while _MADE:
        path = _MADE.pop()
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            left.append(path)
    if WORK.is_dir() and not any(WORK.iterdir()):
        WORK.rmdir()
    return left


def real_config() -> dict[str, Any]:
    """The real config as a mutable dict (a deep copy - tests change it freely)."""
    with CONFIG.open("rb") as handle:
        return copy.deepcopy(tomllib.load(handle))


def python_config() -> dict[str, Any]:
    """The real config with the python engine (for drawers built inside a test process)."""
    cfg = real_config()
    cfg["hebzip"]["engine"] = "python"
    return cfg


def replaced(text: str, *replacements: tuple[str, str]) -> str:
    """``text`` with exact replacements, each of which must match exactly once."""
    for old, new in replacements:
        assert text.count(old) == 1, f"config text {old!r} must occur exactly once"
        text = text.replace(old, new)
    return text


def config_variant(name: str, *replacements: tuple[str, str]) -> Path:
    """A copy of the real config with exact text replacements, in ``tests/_work/<name>/``."""
    text = replaced(CONFIG.read_text(encoding="utf-8"), *replacements)
    path = work_dir(name) / "hebzip.toml"
    path.write_text(text, encoding="utf-8")
    tomllib.loads(text)
    return path


def edit_config(path: Path, *replacements: tuple[str, str]) -> None:
    """Change a variant config in place (e.g. after its derived copy was made from it)."""
    text = replaced(path.read_text(encoding="utf-8"), *replacements)
    tomllib.loads(text)
    path.write_text(text, encoding="utf-8")


def engine_config(engine: str, name: str) -> Path:
    """The real config with ``[hebzip].engine = <engine>``."""
    return config_variant(name, ('engine = "numba"', f'engine = "{engine}"'))


def store_row(text: str) -> str:
    """The ``table_store`` row of a config text exactly as written (what a variant replaces)."""
    return f'table_store = "{tomllib.loads(text)["hebzip"]["table_store"]}"'


STORE_ROW = store_row(CONFIG.read_text(encoding="utf-8"))
"""The real config's store row as written, whatever store the user chose. Why derived: it was hard-coded
"eager", so switching the real config to mmap broke every store variant.
``tests/store/test_table_stores.py`` proves ``store_config`` gives each store from whatever the real row is."""
STORE_DIR_ROW = 'table_store_dir = "tables/mmap"'
STORES = ("eager", "mmap")
"""The two real stores; tests that depend on the store run in both, whatever the real config selects."""


def store_dir(name: str) -> str:
    """The derived copy's folder of the variant ``name`` (relative, as written into its config)."""
    return f"tests/_work/{name}/mm"


def store_config(name: str, store: str, *replacements: tuple[str, str]) -> Path:
    """The real config with ``[hebzip].table_store = <store>``; its derived (mapped) copy lives in
    ``tests/_work/<name>/mm`` - never the real ``tables/mmap``."""
    return config_variant(name, (STORE_ROW, f'table_store = "{store}"'),
                          (STORE_DIR_ROW, f'table_store_dir = "{store_dir(name)}"'), *replacements)


def ready_store_config(name: str, store: str, modes: tuple[str, ...],
                       *replacements: tuple[str, str]) -> Path:
    """``store_config`` that can be used at once: for ``mmap`` the derived copy of ``modes`` is made first
    (the same step a user takes: convert, then compress)."""
    cfg = store_config(name, store, *replacements)
    if store == "mmap":
        done = convert_tables(cfg, *modes)
        assert done.returncode == 0, done.stderr[-800:]
    return cfg


def convert_tables(config: Path, *modes: str) -> subprocess.CompletedProcess[bytes]:
    """The real converter (``python -m hebzip.ops.convert_tables``) in a child process."""
    cmd = [sys.executable, "-m", "hebzip.ops.convert_tables", "--config", str(config)]
    cmd += ["--modes", ",".join(modes)] if modes else []
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=CLI_TIMEOUT_S, check=False)


def compress_file(src: Path, dst: Path, mode: str, *extra: str) -> subprocess.CompletedProcess[bytes]:
    """CLI compress with --force."""
    return run_cli("compress", "--mode", mode, *extra, "--force", src, dst)


def decompress_file(src: Path, dst: Path, *extra: str,
                    timeout: int = CLI_TIMEOUT_S) -> subprocess.CompletedProcess[bytes]:
    """CLI decompress with --force."""
    return run_cli("decompress", *extra, "--force", src, dst, timeout=timeout)


def assert_same_signatures(port: type, *objects: object) -> None:
    """Every public method of ``port`` has the same parameters (names, kinds, annotations) on each object."""
    names = [n for n, v in vars(port).items() if callable(v) and not n.startswith("_")]
    assert names, "the contract has no methods?"
    for name in names:
        wanted = inspect.signature(getattr(port, name))
        for obj in objects:
            got = inspect.signature(getattr(obj, name))
            assert list(got.parameters.values()) == [p for p in wanted.parameters.values() if p.name != "self"], (
                f"{type(obj).__name__}.{name}: {got} != {wanted}")
