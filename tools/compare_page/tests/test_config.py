"""The real config loads; a typo or a wrong value stops the start with a Hebrew sentence."""

from __future__ import annotations

import tomllib
from typing import Any

import pytest

from atar.compressors.loaded import ServicePool
from atar.compressors.registry import build
from atar.config.load import DEFAULT_CONFIG, load, parse
from atar.core.errors import ConfigError


def _raw() -> dict[str, Any]:
    return tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_real_config_has_the_seven_rows_in_run_order() -> None:
    s = load()
    assert s.host == "127.0.0.1" and s.port not in (8795, 8199)
    assert [t.id for t in s.tools] == ["zip", "bz2", "xz", "ppmd", "paq8px", "fast", "max", "max_plus"]
    assert [t.id for t in s.tools if t.ours] == ["fast", "max", "max_plus"]
    pool = ServicePool(s.services)
    for row in s.tools:
        build(row, pool)  # every drawer accepts its options
    assert [x.name for x in s.services] == ["hebzip"]
    assert s.services[0].command[-3:] == ("serve", "--modes", "fast,max,max_plus")


@pytest.mark.parametrize("mutate", [
    lambda d: d["server"].update(prot=1),
    lambda d: d["server"].update(host="0.0.0.0"),
    lambda d: d["limits"].update(max_text_bytes="200"),
    lambda d: d["tools"][0].update(colour="green"),
    lambda d: d["tools"].append(dict(d["tools"][0])),
    lambda d: d.pop("services"),
    lambda d: d["services"]["hebzip"].update(colour="green"),
    lambda d: d["services"]["hebzip"].update(command=[]),
    lambda d: d["services"]["hebzip"].update(ready_timeout_s=0),
    lambda d: d["services"]["hebzip"].pop("cwd"),
])
def test_bad_config_is_refused(mutate: Any) -> None:
    data = _raw()
    mutate(data)
    with pytest.raises(ConfigError):
        parse(data)


@pytest.mark.parametrize(("index", "options"), [
    (0, {"level": 9, "speed": 1}), (2, {"preset": 9}), (3, {"order": "8", "memory_mb": 64}),
    (4, {"version": "x"}),
])
def test_bad_tool_options_are_refused(index: int, options: dict[str, Any]) -> None:
    data = _raw()
    data["tools"][index]["options"] = options
    with pytest.raises(ConfigError):
        build(parse(data).tools[index])


def test_unknown_kind_is_refused() -> None:
    data = _raw()
    data["tools"][0]["kind"] = "gzipp"
    with pytest.raises(ConfigError):
        build(parse(data).tools[0])
