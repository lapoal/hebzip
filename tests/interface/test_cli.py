"""The command line: help, input errors (exit 7), no overwrite, ``info``, damaged input,
the checksum (design decision), and the empty file. All through ``python -m hebzip`` in a child process."""

from __future__ import annotations

from pathlib import Path

from tests.support import GOLDEN, compress_file, config_variant, decompress_file, golden_package, run_cli, work_dir

EXIT_INPUT, EXIT_CORRUPT = 7, 4
RAW_MAX_BYTE0 = 0x4F
"""The first byte of the research prototype's raw max packages (dict 'all' << 4 | variant 15; no header)."""


def _raw_old_package(work: Path) -> Path:
    """A package in the research prototype's raw layout: its one-byte prefix, then a payload, no header."""
    path = work / "raw_old.bin"
    path.write_bytes(bytes([RAW_MAX_BYTE0]) + golden_package("max", "own_letter")[2:])
    return path


def test_help_lists_the_commands() -> None:
    done = run_cli("--help")
    assert done.returncode == 0
    out = done.stdout.decode()
    assert all(word in out for word in ("compress", "decompress", "info"))


def test_missing_input_is_exit_7() -> None:
    work = work_dir("cli_missing")
    done = run_cli("compress", work / "nope.txt", work / "x.hz")
    assert done.returncode == EXIT_INPUT and not (work / "x.hz").exists()


def test_existing_output_is_kept_without_force() -> None:
    work = work_dir("cli_force")
    out = work / "x.hz"
    out.write_bytes(b"precious")
    src = GOLDEN / "texts" / "own_letter.txt"
    done = run_cli("compress", "--mode", "generic", src, out)
    assert done.returncode == EXIT_INPUT and out.read_bytes() == b"precious"
    assert compress_file(src, out, "generic").returncode == 0 and out.read_bytes() != b"precious"


def test_not_utf8_input_is_exit_7() -> None:
    work = work_dir("cli_utf8")
    bad = work / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00 not utf-8 \xc3")
    assert compress_file(bad, work / "x.hz", "fast").returncode == EXIT_INPUT


def test_info_describes_packages() -> None:
    work = work_dir("cli_info")
    pkg = work / "x.hz"
    assert compress_file(GOLDEN / "texts" / "own_letter.txt", pkg, "fast", "--no-fallback").returncode == 0
    new = run_cli("info", pkg).stdout.decode()
    assert "format: v1" in new and "version: 1" in new and "mode: fast" in new and f"size: {pkg.stat().st_size}" in new
    old = run_cli("info", _raw_old_package(work)).stdout.decode()
    assert "format: legacy" in old and "supported: no" in old and "0x4f" in old


def test_raw_experiment_package_is_refused() -> None:
    """Design decision: no decode path for the research prototype's raw v13/v15 packages."""
    work = work_dir("cli_raw_old")
    done = decompress_file(_raw_old_package(work), work / "x.out")
    assert done.returncode == EXIT_CORRUPT and not (work / "x.out").exists()


def test_garbage_and_damaged_inputs_are_typed_errors_not_crashes() -> None:
    work = work_dir("cli_garbage")
    for name, data in (("junk", b"hello world, not a package"), ("empty", b""), ("cut", bytes([0x81]))):
        src = work / name
        src.write_bytes(data)
        done = decompress_file(src, work / f"{name}.out")
        assert done.returncode == EXIT_CORRUPT, (name, done.returncode, done.stderr[-200:])
        assert not (work / f"{name}.out").exists()


def test_checksum_on_detects_a_flipped_payload_byte() -> None:
    work = work_dir("cli_crc")
    cfg = config_variant("cli_crc_cfg", ('integrity = "none"', 'integrity = "crc16"'))
    src, pkg = GOLDEN / "texts" / "own_letter.txt", work / "x.hz"
    assert compress_file(src, pkg, "fast", "--no-fallback", "--config", str(cfg)).returncode == 0
    data = bytearray(pkg.read_bytes())
    assert data[0] == 0xFF  # a checksum needs the extended header
    data[-3] ^= 0x01
    pkg.write_bytes(bytes(data))
    out = work / "x.out"
    done = decompress_file(pkg, out)  # default config: integrity off, but the PACKAGE says it has a checksum
    assert done.returncode == EXIT_CORRUPT and "סכום הביקורת" in done.stderr.decode("utf-8")
    assert not out.exists()


def test_empty_file_round_trips() -> None:
    work = work_dir("cli_empty")
    src, pkg, back = work / "e.txt", work / "e.hz", work / "e.out"
    src.write_bytes(b"")
    assert compress_file(src, pkg, "fast").returncode == 0
    assert decompress_file(pkg, back).returncode == 0 and back.read_bytes() == b""


def test_crlf_bom_and_mixed_text_round_trip_exactly() -> None:
    work = work_dir("cli_bytes")
    src, pkg, back = work / "t.txt", work / "t.hz", work / "t.out"
    src.write_bytes("﻿שלום עולם\r\nשורה שנייה with English 123\r\n".encode())
    assert compress_file(src, pkg, "fast").returncode == 0
    assert decompress_file(pkg, back).returncode == 0 and back.read_bytes() == src.read_bytes()
