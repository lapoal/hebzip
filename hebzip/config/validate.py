"""Row-by-row validation of the config: every bad value names its row.

Why separate from ``load.py``: loading (TOML + shape) and judging (is the value allowed, do the
references exist) change for different reasons. Every message has the form
``<file>: [section].row = value: why`` so the user sees which line of hebzip.toml to fix. Start-up
refuses on the first bad row. Mode rows (drawers and their numbers) are judged in ``validate_model``.
"""

from __future__ import annotations

from pathlib import Path

from hebzip.config import schema as S
from hebzip.config.checks import bad, in_range, one_of
from hebzip.config.validate_model import check_mode
from hebzip.config.validate_recipe import check_recipe

HEX = frozenset("0123456789abcdef")
SHA_LEN = 64
CODE_MIN, CODE_MAX = 0xF8, 0xFF
MAX_CODEPOINT = 0x10FFFF
PHANTOM_RANGE = (128, 1 << 20)
MACHINE_PARTS = ("hebzip", "config", "tables", "tests", ".cache", ".venv")
"""Top folders of the machine: a material folder may not be (inside) one of them."""


def validate(settings: S.Settings, source: str, root: Path) -> None:
    """Raise ``ConfigError`` (naming the row) on the first invalid row; return when all is fine."""
    _check_global(settings, source)
    _check_material(settings.hebzip.material_dir, source, root)
    _check_store(settings.hebzip, source)
    for name, table in settings.tables.items():
        _check_table(name, table, source, root, frozenset(settings.tables))
    for name, lang in settings.languages.items():
        _check_language(name, lang, settings, source)
    ids: dict[int, str] = {}
    for name, row in settings.modes.items():
        check_mode(name, row, settings, source)
        if row.id in ids:
            raise bad(source, f"[modes.{name}].id", row.id, f"already used by [modes.{ids[row.id]}]")
        ids[row.id] = name
    _check_names(settings, source)


def _check_global(settings: S.Settings, source: str) -> None:
    h = settings.hebzip
    if h.format_version not in S.FORMAT_VERSIONS:
        raise bad(source, "[hebzip].format_version", h.format_version, "not one of 1")
    one_of(source, "[hebzip].fallback_policy", h.fallback_policy, S.FALLBACK_POLICIES)
    one_of(source, "[hebzip].integrity", h.integrity, S.INTEGRITY_WORDS)
    one_of(source, "[hebzip].engine", h.engine, S.ENGINE_WORDS)
    one_of(source, "[hebzip].log_level", h.log_level, S.LOG_LEVELS)
    if h.holes not in S.HOLES_WORDS:
        raise bad(source, "[hebzip].holes", h.holes, "holes: only 'null' is built")
    in_range(source, "[hebzip].max_input_bytes", h.max_input_bytes, 1, 2**31)
    if h.max_output_bytes < h.max_input_bytes:
        raise bad(source, "[hebzip].max_output_bytes", h.max_output_bytes, "must be >= max_input_bytes")
    in_range(source, "[hebzip].phantom_bits_max", h.phantom_bits_max, *PHANTOM_RANGE)


def _check_material(material: str, source: str, root: Path) -> None:
    """the machine-and-material rule: the material folder is a real folder inside the project that holds no
    machine part.

    Why inside and apart: the canary gate scans the project minus this folder; a material folder that
    is the project itself, or that holds the code / tables / config, would exempt the machine from the scan.
    """
    row = "[hebzip].material_dir"
    path = Path(material)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise bad(source, row, material, "must be a folder name inside the project (relative, no '..')")
    if path.parts[0] in MACHINE_PARTS:
        raise bad(source, row, material, f"must not be inside the machine ({', '.join(MACHINE_PARTS)})")
    if not (root / path).is_dir():
        raise bad(source, row, material, "folder does not exist")


def _check_store(h: S.HebzipRow, source: str) -> None:
    """The store word, its derived folder and the allowed widths.

    The folder is a relative path inside the project and not inside the material folder - the derived copy
    is machine (tables), never material (the machine-and-material rule). Whether it EXISTS is the mapped store's own
    check at the first table it is asked for (exit 3 naming the converter), so a config that never maps
    still starts without one."""
    one_of(source, "[hebzip].table_store", h.table_store, S.STORE_WORDS)
    path = Path(h.table_store_dir)
    material = Path(h.material_dir)
    if not h.table_store_dir or path.is_absolute() or path.drive or ".." in path.parts:
        raise bad(source, "[hebzip].table_store_dir", h.table_store_dir, "must be a folder inside the project "
                  "(relative, no '..')")
    if path.parts[:len(material.parts)] == material.parts:
        raise bad(source, "[hebzip].table_store_dir", h.table_store_dir, "must not be inside the material folder")
    bits = h.narrow_bits
    if not bits or any(b not in S.NARROW_WIDTHS for b in bits) or bits != sorted(set(bits)):
        raise bad(source, "[hebzip].narrow_bits", bits, f"rising, no repeats, each one of {list(S.NARROW_WIDTHS)}")


def _check_table(name: str, table: S.TableFile, source: str, root: Path, names: frozenset[str]) -> None:
    where = f"[tables.{name}]"
    if len(table.sha256) != SHA_LEN or not set(table.sha256) <= HEX:
        raise bad(source, f"{where}.sha256", table.sha256, "must be 64 lowercase hex characters")
    if table.bytes < 0:
        raise bad(source, f"{where}.bytes", table.bytes, "must be >= 0")
    shape = Path(table.path)
    if not table.path or shape.is_absolute() or shape.drive or ".." in shape.parts:
        raise bad(source, f"{where}.path", table.path, "must be a relative path inside the project (no '..')")
    # Whether the file EXISTS is the table store's check when a mode first needs it (``store/missing.py``):
    # the tables are release downloads, so a fresh clone with only some of them (the four of
    # mode fast from site/tables) must still start and run those modes; a missing file then ends in
    # ``TablesMissing`` (exit 3) saying where to get it, instead of refusing every mode at start-up.
    if table.recipe is not None:
        check_recipe(f"[tables.{name}.recipe]", table.recipe, source, names)


def resolve(root: Path, path: str) -> Path:
    """A table path: absolute as written, relative = under the project root."""
    p = Path(path)
    return p if p.is_absolute() else root / p


def _check_language(name: str, lang: S.LanguageRow, settings: S.Settings, source: str) -> None:
    at = f"[languages.{name}]"
    for row in ("letters", "marks"):
        for i, pair in enumerate(getattr(lang, row)):
            if len(pair) != 2 or not 1 <= pair[0] <= pair[1] <= MAX_CODEPOINT:
                raise bad(source, f"{at}.{row}[{i}]", pair, "must be [first, last] code points, first <= last")
    codes = [lang.slot_plain, lang.slot_marked, lang.mark_separator, lang.marked_end]
    for row, value in zip(("slot_plain", "slot_marked", "mark_separator", "marked_end"), codes, strict=True):
        in_range(source, f"{at}.{row}", value, CODE_MIN, CODE_MAX)
    if len(set(codes)) != len(codes):
        raise bad(source, f"{at}.slot_plain", lang.slot_plain, "the four template codes must differ")
    if lang.word_list not in settings.tables:
        raise bad(source, f"{at}.word_list", lang.word_list, "no such [tables.*]")
    for i, mark in enumerate(lang.sentence_end):
        if len(mark) != 1:
            raise bad(source, f"{at}.sentence_end[{i}]", mark, "must be one character")
    for i, marks in enumerate(lang.gap_marks):
        if not marks:
            raise bad(source, f"{at}.gap_marks[{i}]", marks, "an empty gap class (give it characters)")
    if len(set(lang.prefixes)) != len(lang.prefixes):
        raise bad(source, f"{at}.prefixes", lang.prefixes, "each proclitic letter once")


def _check_names(settings: S.Settings, source: str) -> None:
    h = settings.hebzip
    if h.default_mode not in settings.modes:
        raise bad(source, "[hebzip].default_mode", h.default_mode, "no such [modes.*]")
    if settings.modes[h.default_mode].retired:
        raise bad(source, "[hebzip].default_mode", h.default_mode, "is retired (decoding only)")
    if len(set(h.fallback)) != len(h.fallback):
        raise bad(source, "[hebzip].fallback", h.fallback, "duplicate names")
    for name in h.fallback:
        row = settings.modes.get(name)
        if row is None or not S.FALLBACK_KINDS[row.kind]:
            raise bad(source, "[hebzip].fallback", name, "must name a mode with kind = 'generic'")
        if row.retired:
            #: a retired fallback used to compress silently (retired = decode only)
            raise bad(source, "[hebzip].fallback", name, "is retired (decoding only)")
