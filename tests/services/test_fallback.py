"""Fallback to the generic codec, through the real CLI.

* ``smaller``: the output is the smaller of primary/generic (English text makes the choice real).
* ``on_error``: the primary stays even when bigger; only a refusing drawer (``DrawerUnavailable``) falls back.
* ``fallback = []`` (or ``--no-fallback``) + a refusing drawer -> exit 6. The refusing mode is ``max``
  with its word model switched to the plate (one word in a config copy; max itself works).
"""

from __future__ import annotations

from pathlib import Path

from hebzip.domain.header import parse_header
from tests.support import GOLDEN, compress_file, config_variant, decompress_file, work_dir

EXIT_UNAVAILABLE = 6
ENGLISH = ("The quick brown fox jumps over the lazy dog. Packing my box with five dozen liquor jugs. " * 24)
REFUSING = "max"
PLATE_WORD = ('word_model = "mixture"\nword_coder = "spelled"           #',
              'word_model = "null"\nword_coder = "spelled"           #')
"""max's word model switched to the plate: it refuses at the first word."""


def _config(name: str, policy: str = "smaller", fallback: str = '["generic"]') -> Path:
    return config_variant(name, ('fallback_policy = "smaller"', f'fallback_policy = "{policy}"'),
                          ('fallback = ["generic"]', f"fallback = {fallback}"), PLATE_WORD)


def _mode_id(path: Path) -> int:
    return parse_header(path.read_bytes()).mode_id


def test_smaller_policy_keeps_the_smaller_package() -> None:
    work = work_dir("fallback_smaller")
    src = work / "english.txt"
    src.write_bytes(ENGLISH.encode())
    sizes = {}
    for label, mode, extra in (("fast", "fast", ("--no-fallback",)), ("generic", "generic", ("--no-fallback",)),
                               ("auto", "fast", ())):
        out = work / f"{label}.hz"
        assert compress_file(src, out, mode, *extra).returncode == 0
        sizes[label] = out.stat().st_size
    assert sizes["auto"] == min(sizes["fast"], sizes["generic"])
    back = work / "back.txt"
    assert decompress_file(work / "auto.hz", back).returncode == 0 and back.read_bytes() == src.read_bytes()
    hebrew = GOLDEN / "texts" / "own_letter.txt"
    assert compress_file(hebrew, work / "he.hz", "fast").returncode == 0
    assert _mode_id(work / "he.hz") == 1  # Hebrew: the primary wins, the fallback is not used


def test_on_error_policy_keeps_the_primary_even_when_bigger() -> None:
    work = work_dir("fallback_on_error")
    cfg = _config("fallback_on_error_cfg", policy="on_error")
    src = work / "english.txt"
    src.write_bytes(ENGLISH.encode())
    out = work / "x.hz"
    assert compress_file(src, out, "fast", "--config", str(cfg)).returncode == 0
    assert _mode_id(out) == 1


def test_refusing_drawer_falls_back_to_generic() -> None:
    for policy in ("on_error", "smaller"):
        work = work_dir(f"fallback_dead_{policy}")
        cfg = _config(f"fallback_dead_cfg_{policy}", policy=policy)
        src = GOLDEN / "texts" / "own_letter.txt"
        out, back = work / "x.hz", work / "x.out"
        assert compress_file(src, out, REFUSING, "--config", str(cfg)).returncode == 0
        assert _mode_id(out) == 3
        assert decompress_file(out, back, "--config", str(cfg)).returncode == 0
        assert back.read_bytes() == src.read_bytes()


def test_no_fallback_and_a_refusing_drawer_is_exit_6() -> None:
    work = work_dir("fallback_none")
    src = GOLDEN / "texts" / "own_letter.txt"
    off = _config("fallback_none_cfg", fallback="[]")
    done = compress_file(src, work / "a.hz", REFUSING, "--config", str(off))
    assert done.returncode == EXIT_UNAVAILABLE and not (work / "a.hz").exists()
    plate = _config("fallback_none_plate_cfg")
    flag = compress_file(src, work / "b.hz", REFUSING, "--no-fallback", "--config", str(plate))
    assert flag.returncode == EXIT_UNAVAILABLE and not (work / "b.hz").exists()
