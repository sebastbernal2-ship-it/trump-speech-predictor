from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/idea_layers/relations/extract_layered_expression_anchors.py"
SPEC = importlib.util.spec_from_file_location("extract_layered_expression_anchors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
anchors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchors)


def test_anchor_response_parser_requires_exact_two_layers():
    parsed = anchors._parse_result('{"anchors":[{"mention_id":"m1","idea_id":"i1"}],"unanchored_idea_ids":[]}')
    assert parsed == ([{"mention_id": "m1", "idea_id": "i1"}], [])
    with pytest.raises(ValueError, match="exactly anchors"):
        anchors._parse_result('{"anchors":[],"unanchored_idea_ids":[],"relations":[]}')


def test_anchor_prompt_hides_legacy_links_and_forecastability():
    prompt = anchors._prompt(
        [{"turn_id": "blind:turn:0000", "text": "The policy should lower costs."}],
        {
            "mentions": [{"mention_id": "m1", "text": "The policy lowers costs.", "evidence": [
                {"turn_id": "source:turn:1", "exact_quote": "The policy should lower costs."},
            ]}],
            "ideas": [{"idea_id": "i1", "summary": "The policy should lower costs.", "forecastable": True}],
            "links": [{"mention_id": "m1", "idea_id": "i1", "relation": "legacy_relation_must_not_be_seen"}],
        },
        {"source:turn:1": "blind:turn:0000"},
    )
    assert "legacy_relation_must_not_be_seen" not in prompt
    assert '"forecastable"' not in prompt
    assert "unanchored_idea_ids" in prompt
