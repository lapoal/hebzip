"""Config validation of the global rows, tables and language pack: one invalid value
per row -> ``ConfigError`` naming that row; unknown fields at every level; the real file loads.
Mode and drawer rows: ``test_load_model.py``."""

from __future__ import annotations

from typing import Any

import pytest

from hebzip.config.load import load_config, load_dict
from hebzip.core.errors import ConfigError
from tests.support import CONFIG, real_config

Path = tuple[str | int, ...]
BAD: list[tuple[Path, Any, str]] = [
    (("hebzip", "format_version"), 2, "[hebzip].format_version"),
    (("modes", "fast", "language"), "xx", "[modes.fast].language"),   # not "en" - that is a real pack
    (("hebzip", "default_mode"), "nope", "[hebzip].default_mode"),
    (("hebzip", "fallback"), ["fast"], "[hebzip].fallback"),
    (("hebzip", "fallback"), ["generic", "generic"], "[hebzip].fallback"),
    (("hebzip", "fallback"), ["ghost"], "[hebzip].fallback"),
    (("hebzip", "fallback_policy"), "always", "[hebzip].fallback_policy"),
    (("hebzip", "verify_roundtrip"), "yes", "verify_roundtrip"),
    (("hebzip", "integrity"), "md5", "[hebzip].integrity"),
    (("hebzip", "engine"), "legacy", "[hebzip].engine"),
    (("hebzip", "holes"), "decoder", "only 'null' is built"),
    (("hebzip", "signature_cache"), 5, "signature_cache"),
    # the machine-and-material rule: the one material folder - real, inside the project, apart from the machine
    (("hebzip", "material_dir"), "no_such_folder", "[hebzip].material_dir"),
    (("hebzip", "material_dir"), "tables", "[hebzip].material_dir"),
    (("hebzip", "material_dir"), "hebzip/domain", "[hebzip].material_dir"),
    (("hebzip", "material_dir"), ".", "[hebzip].material_dir"),
    (("hebzip", "material_dir"), "../golden", "[hebzip].material_dir"),
    (("hebzip", "material_dir"), "C:/golden", "[hebzip].material_dir"),
    # the table store - a word, its derived folder inside the project (not the material), widths
    (("hebzip", "table_store"), "lazy", "[hebzip].table_store"),
    (("hebzip", "table_store_dir"), "../mm", "[hebzip].table_store_dir"),
    (("hebzip", "table_store_dir"), "C:/mm", "[hebzip].table_store_dir"),
    (("hebzip", "table_store_dir"), "", "[hebzip].table_store_dir"),
    (("hebzip", "table_store_dir"), "golden/mm", "[hebzip].table_store_dir"),
    (("hebzip", "narrow_bits"), [], "[hebzip].narrow_bits"),
    (("hebzip", "narrow_bits"), [16, 8], "[hebzip].narrow_bits"),
    (("hebzip", "narrow_bits"), [8, 8, 64], "[hebzip].narrow_bits"),
    (("hebzip", "narrow_bits"), [8, 12], "[hebzip].narrow_bits"),
    (("hebzip", "max_input_bytes"), 0, "[hebzip].max_input_bytes"),
    (("hebzip", "max_output_bytes"), 10, "[hebzip].max_output_bytes"),
    (("hebzip", "phantom_bits_max"), 10, "[hebzip].phantom_bits_max"),
    (("hebzip", "phantom_bits_max"), 1 << 30, "[hebzip].phantom_bits_max"),
    (("hebzip", "log_level"), "TRACE", "[hebzip].log_level"),
    (("tables", "he_words", "sha256"), "ab12", "[tables.he_words].sha256"),
    (("tables", "he_words", "path"), "../outside/words.txt", "[tables.he_words].path"),  # a missing file is
    # refused when a mode needs it (tests/store/test_tables_missing.py), not at start-up
    (("tables", "he_words", "bytes"), -1, "[tables.he_words].bytes"),
    (("languages", "he", "letters"), [[0x05EA, 0x05D0]], "[languages.he].letters[0]"),
    (("languages", "he", "marks"), [[0x0591]], "[languages.he].marks[0]"),
    (("languages", "he", "slot_plain"), 0x41, "[languages.he].slot_plain"),
    (("languages", "he", "marked_end"), 0xFF, "the four template codes must differ"),
    (("languages", "he", "word_list"), "ghost", "[languages.he].word_list"),
    (("languages", "he", "sentence_end"), ["ab"], "[languages.he].sentence_end[0]"),
    (("languages", "he", "gap_marks"), [",;", ""], "[languages.he].gap_marks[1]"),
    (("modes", "generic", "retired"), True, "[hebzip].fallback"),
    (("modes", "fast", "retired"), True, "[hebzip].default_mode"),
    # the language is a row of each model mode (a generic mode has none; the global row is gone)
    (("hebzip", "language"), "he", "language"),
    (("modes", "generic", "language"), "he", "[modes.generic].language"),
    (("languages", "he", "prefixes"), "ווה", "[languages.he].prefixes"),
    # table recipes are rows, checked at start-up like every other row
    (("tables", "template_m128", "recipe", "kind"), "magic", "[tables.template_m128.recipe].kind"),
    (("tables", "template_m128", "recipe", "min_count"), 0, "[tables.template_m128.recipe].min_count"),
    (("tables", "template_m128", "recipe", "count_levels"), [4, 2], "[tables.template_m128.recipe].count_levels"),
    (("tables", "template_m128", "recipe", "keep_top"), 5, "not used by kind = 'template_prior'"),
    (("tables", "prefix_classes", "recipe", "double_first"), "וו", "[tables.prefix_classes.recipe].double_first"),
    (("tables", "template_m128_weights", "recipe", "held_every"), -1,
     "[tables.template_m128_weights.recipe].held_every"),
]
MISSING: list[tuple[Path, str]] = [
    (("hebzip", "engine"), "engine"),
    (("hebzip", "phantom_bits_max"), "phantom_bits_max"),
    (("hebzip", "material_dir"), "material_dir"),
    (("hebzip", "table_store"), "table_store"),
    (("hebzip", "table_store_dir"), "table_store_dir"),
    (("hebzip", "narrow_bits"), "narrow_bits"),
    (("languages", "he", "word_list"), "word_list"),
    (("tables", "he_words", "sha256"), "sha256"),
    (("modes", "max", "language"), "[modes.max].language"),
    (("tables", "template_m128_weights", "recipe", "fit_cut"), "[tables.template_m128_weights.recipe].fit_cut"),
]


def set_path(cfg: dict[str, Any], path: Path, value: Any) -> None:
    """Set one value deep inside a parsed config."""
    node: Any = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def refused(cfg: dict[str, Any]) -> str:
    """The ConfigError message for this config (fails if it loads)."""
    with pytest.raises(ConfigError) as caught:
        load_dict(cfg)
    return str(caught.value)


@pytest.mark.parametrize(("path", "value", "needle"), BAD, ids=[f"{'.'.join(map(str, p))}" for p, _, _ in BAD])
def test_invalid_row_is_refused_and_named(path: Path, value: Any, needle: str) -> None:
    cfg = real_config()
    set_path(cfg, path, value)
    assert needle in refused(cfg)


@pytest.mark.parametrize(("path", "needle"), MISSING, ids=[".".join(map(str, p)) for p, _ in MISSING])
def test_missing_required_row_is_refused(path: Path, needle: str) -> None:
    cfg = real_config()
    node: Any = cfg
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    assert needle in refused(cfg)


def test_unknown_field_at_every_level() -> None:
    places = ((), ("hebzip",), ("tables", "he_words"), ("languages", "he"), ("modes", "fast"),
              ("modes", "fast", "template"), ("modes", "fast", "base"), ("modes", "fast", "mixer"),
              ("modes", "fast", "predictors", "text_pairs"), ("modes", "generic"), ("modes", "max", "gap"),
              ("modes", "max", "speller"), ("modes", "max", "letter_chains", "cross"),
              ("modes", "max", "letters", "text_letters"), ("modes", "max", "letter_mixer"),
              ("modes", "max", "predictors", "prefix_class"), ("tables", "template_m128", "recipe"))
    for where in places:
        cfg = real_config()
        node: Any = cfg
        for key in where:
            node = node[key]
        node["surprise"] = 1
        assert "surprise" in refused(cfg), where


def test_the_example_message() -> None:
    cfg = real_config()
    cfg["modes"]["max"]["word_model"] = "letters_v16"
    with pytest.raises(ConfigError) as caught:
        load_dict(cfg, "config\\hebzip.toml")
    assert str(caught.value) == 'config\\hebzip.toml: [modes.max].word_model = "letters_v16": not one of mixture, null'


def test_real_file_loads() -> None:
    settings = load_config(CONFIG)
    assert settings.hebzip.integrity == "none" and settings.hebzip.fallback == ["generic"]
    assert {n: r.id for n, r in settings.modes.items()} == {"fast": 1, "max": 2, "generic": 3, "latin_fast": 4,
                                                            "latin_max": 5, "en_fast": 6, "en_max": 7, "max_plus": 8}
    assert {n: r.language for n, r in settings.modes.items()} == {
        "fast": "he", "max": "he", "generic": None, "latin_fast": "latin_toy", "latin_max": "latin_toy",
        "en_fast": "en", "en_max": "en", "max_plus": "he"}
    assert settings.modes["fast"].channels == ["corpus_pairs", "text_pairs", "text_counts"]
    he = {"he_words", "bigram_mc1", "template_m128", "template_m128_weights", "corpus_gap_starts", "prefix_classes",
          "prefix_prior", "trie_he", "letters_v15", "letter_start_sets"}
    plus_tables = {"word_classes", "class_pairs_wiki", "trust_sets_wiki", "topic_neighbours", "topic_trust_sets"}
    assert {n for n in settings.tables if not n.startswith(("toy_", "en_"))} == he | plus_tables
    assert settings.modes["max"].word_coder == "spelled" and len(settings.modes["max"].channels) == 8
    assert all(not row.path.startswith(("C:", "/")) for row in settings.tables.values())


def test_missing_file_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIG.with_name("nope.toml"))
