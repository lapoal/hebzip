"""Canary gate (the machine-and-material rule): after a real run no piece of the text
is left anywhere outside the declared material folder.

How: a text carrying a fresh random marker (Hebrew letters + Latin) goes through the real CLI, compress
and decompress in separate processes, modes fast, max and max_plus - and through the loaded service (``hebzip serve``),
all three in one process, whose stdout/stderr are scanned too. Its own folder (input, packages, decoded
copies) is made in the system temp folder and deleted at the end. Then we look for the marker in:
- every stdout / stderr the runs printed (logs and error messages never hold content);
- the project tree minus ``[hebzip].material_dir``: every file in ``.cache`` and ``tests/_work``, and
  every other file created or changed during the test (``.venv`` included - a before/after snapshot of
  file stamps, so no clock is read and only changed files are read);
- the system temp folder: top-level files new or changed during the test, and every new top-level folder.
  (Known limit: a file dropped deep inside an OLD temp sub-folder is not seen.)
The marker is searched as UTF-8 and UTF-16 bytes. A witness check first proves the scanner does find
the marker in the input and the decoded files, so a green result is not a blind scanner.

Why: the code, tables, caches and logs were clean, but test scratch folders and
earlier scratch folders kept decoded copies and packages of private texts - and no gate said so.
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from hebzip.config.load import load_config
from tests.serve_client import Service
from tests.support import CONFIG, ROOT, WORK, run_cli

LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
MODES = ("fast", "max", "max_plus")
ALWAYS_READ = (ROOT / ".cache", WORK)
"""Folders read completely, whatever the file stamps say."""


def _marker() -> tuple[str, str]:
    """A fresh (Hebrew, Latin) marker pair - never seen on this disk before."""
    hebrew = "".join(secrets.choice(LETTERS) for _ in range(14))
    return hebrew, f"canary{secrets.token_hex(8)}"


def _text(hebrew: str, latin: str) -> str:
    return (f"זהו טקסט בדיקה קצר. המילה {hebrew} מופיעה כאן, וגם {latin} באמצע.\n"
            f"שורה שנייה: שוב {hebrew} ושוב {latin}, ואז נקודה.\n")


def _needles(hebrew: str, latin: str) -> list[bytes]:
    """Byte forms of the marker; a 10-letter piece of the Hebrew part is enough to count as a leak."""
    return [part.encode(enc) for part in (hebrew[:10], latin) for enc in ("utf-8", "utf-16-le")]


def _has(data: bytes, needles: list[bytes]) -> bool:
    return any(n in data for n in needles)


def _file_has(path: Path, needles: list[bytes]) -> bool:
    try:
        return _has(path.read_bytes(), needles)
    except OSError:
        return False


Stamps = dict[Path, tuple[int, int, int]]


def _stamp(path: Path) -> tuple[int, int, int] | None:
    """(size, mtime_ns, ctime_ns) - a file's stamp; no clock is read (the one-clock rule)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _tree_files(root: Path, skip: Path | None = None) -> Iterator[Path]:
    for folder, dirs, files in os.walk(root):
        here = Path(folder)
        dirs[:] = [d for d in dirs if skip is None or here / d != skip]
        yield from (here / f for f in files)


def _snapshot(material: Path, temp: Path) -> tuple[Stamps, set[str]]:
    """Stamps of every project file (minus the material) and of the temp folder's top-level files,
    plus the names of the temp folder's top-level entries."""
    stamps: Stamps = {}
    files = list(_tree_files(ROOT, material)) + [p for p in temp.iterdir() if p.is_file()]
    for path in files:
        stamp = _stamp(path)
        if stamp is not None:
            stamps[path] = stamp
    return stamps, {p.name for p in temp.iterdir()}


def _candidates(material: Path, temp: Path, before: Stamps, names: set[str]) -> list[Path]:
    """Files new or changed since the snapshot, everything in ``.cache`` / ``tests/_work``, and every
    file inside a top-level temp folder that did not exist before."""
    out = [p for p in _tree_files(ROOT, material)
           if any(p.is_relative_to(a) for a in ALWAYS_READ) or before.get(p) != _stamp(p)]
    for entry in temp.iterdir():
        if entry.name not in names and entry.is_dir():
            out += list(_tree_files(entry))
        elif entry.is_file() and before.get(entry) != _stamp(entry):
            out.append(entry)
    return out


def _run_all(folder: Path, source: Path) -> list[tuple[str, bytes]]:
    """Compress + decompress in every mode (separate processes); returns (what, printed bytes)."""
    printed: list[tuple[str, bytes]] = []
    for mode in MODES:
        pkg, back = folder / f"{mode}.hz", folder / f"{mode}.out"
        first = run_cli("compress", "--mode", mode, "--force", source, pkg)
        second = run_cli("decompress", "--force", pkg, back)
        assert first.returncode == 0 and second.returncode == 0, (mode, first.returncode, second.returncode)
        assert back.read_bytes() == source.read_bytes(), f"{mode}: the round trip changed the text"
        printed += [(f"compress {mode}", first.stdout + first.stderr),
                    (f"decompress {mode}", second.stdout + second.stderr)]
    return printed + _run_service(folder, source)


def _run_service(folder: Path, source: Path) -> list[tuple[str, bytes]]:
    """The same marker text through the loaded service (``hebzip serve``): one process, every
    mode, compress + decompress; afterwards its stdout and stderr are scanned like the CLI's, and the
    file / temp scan below covers whatever it wrote. Its packages and decoded copies are left in ``folder``
    (``.srv`` / ``.sout``) so the witness check proves the scanner sees them."""
    with Service(*MODES) as service:
        assert service.ready.get("ready") is True, service.ready
        for mode in MODES:
            pkg, back = folder / f"{mode}.srv", folder / f"{mode}.sout"
            assert service.compress(source, pkg, mode)["ok"] and service.decompress(pkg, back)["ok"], mode
            assert back.read_bytes() == source.read_bytes(), f"service {mode}: the round trip changed the text"
    assert service.proc.returncode == 0
    return [("service stdout", bytes(service.out)), ("service stderr", bytes(service.err))]


def test_no_piece_of_the_text_is_left_outside_the_material_folder() -> None:
    material = ROOT / load_config(CONFIG).hebzip.material_dir
    temp = Path(tempfile.gettempdir())
    before, names = _snapshot(material, temp)
    hebrew, latin = _marker()
    needles = _needles(hebrew, latin)
    own = Path(tempfile.mkdtemp(prefix="hebzip_canary_"))
    try:
        source = own / "canary.txt"
        source.write_text(_text(hebrew, latin), encoding="utf-8")
        printed = _run_all(own, source)
        witness = [p.name for p in own.iterdir() if p.suffix in (".txt", ".out", ".sout") and _file_has(p, needles)]
        assert len(witness) == 1 + 2 * len(MODES), f"the scanner is blind: found the marker only in {witness}"
    finally:
        shutil.rmtree(own, ignore_errors=True)
    assert not own.exists(), "the canary's own folder could not be removed"
    leaks = [what for what, data in printed if _has(data, needles)]
    leaks += [str(p) for p in _candidates(material, temp, before, names) if _file_has(p, needles)]
    assert not leaks, f"text left outside the material folder ({len(leaks)}): {leaks[:10]}"
