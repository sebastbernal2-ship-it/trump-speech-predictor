from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "research/idea_layers/stories/speech_news_circulation_case_study.py"
SPEC = importlib.util.spec_from_file_location("speech_news_circulation_case_study", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_term_plan_validates_complete_ids_and_normalizes_terms():
    plan = module.validate_term_plan(
        {"terms_by_idea": [
            {"idea_id": "idea_a", "terms": [" Matt Bevin ", "matt   bevin"]},
            {"idea_id": "idea_b", "terms": ["impeachment inquiry"]},
        ], "prompt_version": "v2"},
        {"idea_a", "idea_b"},
    )
    assert plan == {"idea_a": ["matt bevin"], "idea_b": ["impeachment inquiry"]}
    with pytest.raises(ValueError, match="missing IDs"):
        module.validate_term_plan({"terms_by_idea": [{"idea_id": "idea_a", "terms": ["term"]}]}, {"idea_a", "idea_b"})


def test_search_regex_matches_phrase_boundaries_not_substrings():
    import re

    pattern = module._regex(["impeachment inquiry", "fake news"])
    assert re.search(pattern, "House impeachment inquiry grows")
    assert re.search(pattern, "President talks about fake news")
    assert not re.search(pattern, "irrelevant impeachment inquiries")


def test_exposure_adjusted_burst_detection_finds_spike_without_a_fixed_window():
    counts = [1] * 90 + [35] * 8 + [1] * 90
    exposures = [1000] * len(counts)
    states = module.poisson_burst_states(counts, exposures)
    assert max(states) >= 3
    assert sum(state > 0 for state in states[90:98]) >= 6
    assert sum(state > 0 for state in states[:90]) < 10
    assert sum(state > 0 for state in states[98:]) < 10


def test_burst_detector_rejects_invalid_exposure_and_handles_no_matches():
    with pytest.raises(ValueError, match="exposures positive"):
        module.poisson_burst_states([1, 0], [1, 0])
    assert module.poisson_burst_states([0, 0], [10, 20]) == [0, 0]
