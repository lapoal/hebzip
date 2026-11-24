"""Rows and tables are guarded, all through the real CLI.

(a) a package whose header signature byte is flipped -> ``TableMismatch`` (exit 3), nothing written;
(b) a config with a wrong sha256 for a table -> start-up refuses, naming the table;
(c) the signature covers the ROWS too: a package made with the default rows is refused by a config
    whose one bit-shaping number differs (otherwise it would decode to garbage without any error).
"""

from __future__ import annotations

import pytest

from tests.support import (
    GOLDEN,
    ROOT,
    SIGNATURE_BYTE,
    STORES,
    compress_file,
    config_variant,
    decompress_file,
    edit_config,
    ready_store_config,
    work_dir,
)

EXIT_MISMATCH = 3
LATIN_SAMPLE = ROOT / "langpacks" / "latin_toy" / "samples" / "bridge.txt"
EN_SAMPLE = ROOT / "golden" / "texts_en" / "own_en_note.txt"
SIGNED_TEXT = {"latin_fast": LATIN_SAMPLE, "latin_max": LATIN_SAMPLE, "en_fast": EN_SAMPLE, "en_max": EN_SAMPLE}
"""The text each mode compresses for the signature-byte test (a Hebrew text unless listed)."""


@pytest.mark.parametrize("mode", sorted(SIGNATURE_BYTE))
def test_every_mode_writes_its_expected_signature_byte(mode: str) -> None:
    """Design decision: the template precision rows are signed plainly, always - there is no
    "leave it out when it equals format 1" case. The byte each mode writes is pinned."""
    work = work_dir(f"sig_byte_{mode}")
    text = SIGNED_TEXT.get(mode, GOLDEN / "texts" / "own_letter.txt")
    pkg = work / "x.hz"
    done = compress_file(text, pkg, mode, "--no-fallback")
    assert done.returncode == 0, done.stderr[-300:]
    assert pkg.read_bytes()[1] == SIGNATURE_BYTE[mode], f"{pkg.read_bytes()[1]:#04x}"
WRONG = "0" * 64
BIGRAM_SHA = "1f380ab97e33a5381bf29f5b860cff76cee52e69a25cec18e606f342b854f0d8"


def test_flipped_header_signature_is_refused_and_writes_nothing() -> None:
    work = work_dir("mismatch_header")
    text = GOLDEN / "texts" / "own_letter.txt"
    for mode in ("fast", "generic"):
        pkg, out = work / f"{mode}.hz", work / f"{mode}.out"
        assert compress_file(text, pkg, mode, "--no-fallback").returncode == 0
        data = bytearray(pkg.read_bytes())
        data[1] ^= 0xFF  # the compact header's signature byte
        pkg.write_bytes(bytes(data))
        done = decompress_file(pkg, out)
        assert done.returncode == EXIT_MISMATCH, done.stderr
        assert "טבלאות אחרות" in done.stderr.decode("utf-8")
        assert not out.exists()


@pytest.mark.parametrize("store", STORES)
def test_wrong_table_sha_in_config_refuses_to_start_and_names_the_table(store: str) -> None:
    """In both stores. The derived copy is made from the right
    rows first, then the row is broken - as when the user mistypes a sha256 after converting."""
    work = work_dir(f"mismatch_table_{store}")
    cfg = ready_store_config(f"mismatch_table_cfg_{store}", store, ("fast",))
    edit_config(cfg, (BIGRAM_SHA, WRONG))
    out = work / "x.hz"
    done = compress_file(GOLDEN / "texts" / "own_letter.txt", out, "fast", "--config", str(cfg))
    assert done.returncode == EXIT_MISMATCH, done.stderr[-300:]
    message = done.stderr.decode("utf-8")
    assert "bigram_mc1" in message and "טבלה" in message, message
    assert not out.exists()


def test_a_changed_model_row_refuses_old_packages() -> None:
    work = work_dir("mismatch_row")
    text = GOLDEN / "texts" / "own_letter.txt"
    pkg, out = work / "x.hz", work / "x.out"
    assert compress_file(text, pkg, "fast", "--no-fallback").returncode == 0
    # max has the same row: take fast's by what follows it
    fast_row = "discount = 0.8522727272727273\ntrust0 = 1.0\n\n[modes.fast.predictors.text_pairs]"
    cfg = config_variant("mismatch_row_cfg", (fast_row, fast_row.replace("0.8522727272727273", "0.75")))
    done = decompress_file(pkg, out, "--config", str(cfg))
    assert done.returncode == EXIT_MISMATCH, done.stderr[-300:]
    assert not out.exists()


def test_a_changed_language_row_refuses_old_packages() -> None:
    """The language moved from a global row to a row of each mode; its pack's rows
    must still be inside the signature - a package is never decoded with other language rows."""
    work = work_dir("mismatch_language")
    text = GOLDEN / "texts" / "own_letter.txt"
    pkg, out = work / "x.hz", work / "x.out"
    assert compress_file(text, pkg, "fast", "--no-fallback").returncode == 0
    cfg = config_variant("mismatch_language_cfg", ('prefixes = "והבכלמש"', 'prefixes = "והבכלמ"'))
    done = decompress_file(pkg, out, "--config", str(cfg))
    assert done.returncode == EXIT_MISMATCH, done.stderr[-300:]
    assert not out.exists()
