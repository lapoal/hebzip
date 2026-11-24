"""The topic channel of max_plus through the real CLI: separate processes, exact round trips.

The topic channel switched to its plate (ONE word) gives EXACTLY the payload of the same mode written without
the channel at all (channel list, its rows and its appended start trust removed): the channel is one drawer and
nothing else moves - the appended trust is dropped together with the plated channel. With the channel on, the
payload differs (it is in the path).
"""

from __future__ import annotations

from tests.services.test_plus import Packs, _all, plate_config
from tests.support import config_variant

TOPIC_ROWS = ('[modes.max_plus.predictors.topic]\nkind = "text_topic"\ntable = "topic_neighbours"\nwindow = 64\n'
              'confidence_k = 4.0\ntrust0 = 0.3\n\n')
WITHOUT_TOPIC = (('"recency", "class_pairs", "class_skip", "topic"]', '"recency", "class_pairs", "class_skip"]'),
                 (TOPIC_ROWS, ""), ('trust_sets = "topic_trust_sets"', 'trust_sets = "trust_sets_wiki"'))
"""The max_plus rows written without the topic channel."""


def _payloads(got: Packs) -> dict[str, bytes]:
    errors = [e for _, e in got.values() if e]
    assert not errors, errors
    return {stem: data[2:] for stem, (data, _) in got.items()}


def test_the_topic_plate_gives_exactly_the_mode_without_the_channel() -> None:
    plated = _payloads(_all("topic_plated", "max_plus", plate_config("modes.max_plus.predictors.topic", "null")))
    without = _payloads(_all("topic_without", "max_plus", config_variant("topic_without_cfg", *WITHOUT_TOPIC)))
    real = _payloads(_all("topic_real", "max_plus"))
    assert plated == without, "plate = the mode without the drawer, byte for byte"
    assert any(real[s] != without[s] for s in real), "the topic channel is in the path"
