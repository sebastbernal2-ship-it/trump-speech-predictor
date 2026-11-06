from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "research/extract_layered_transcript_outcomes.py"
SPEC = importlib.util.spec_from_file_location("extract_layered_transcript_outcomes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)
from speechprefix.idea_layer.transcript_outcomes import (
    LayeredOutcomeError,
    idea_targets,
    mentions_for_idea,
    validate_layered_outcome,
)


def _record() -> tuple[dict, dict[str, str]]:
    turns = {
        "turn-1": "The speaker supports this policy and asks Congress to act.",
        "turn-2": "This local project is one example of that broader policy.",
    }
    record = {
        "schema_version": 1,
        "artifact_type": "layered-transcript-outcome",
        "event_id": "event-1",
        "mentions": [
            {
                "mention_id": "policy_request",
                "text": "The speaker supports a policy and urges congressional action.",
                "role": "core_expression",
                "salience": "central",
                "evidence": [{"turn_id": "turn-1", "exact_quote": "supports this policy and asks Congress to act"}],
            },
            {
                "mention_id": "local_project_example",
                "text": "The speaker cites a local project as an example.",
                "role": "illustrative_example",
                "salience": "secondary",
                "evidence": [{"turn_id": "turn-2", "exact_quote": "This local project is one example"}],
            },
        ],
        "ideas": [
            {
                "idea_id": "policy_and_congressional_action",
                "concept": "Policy and congressional action",
                "summary": "The speaker supports the policy and urges Congress to act.",
                "kind": "policy",
                "stance": "support",
                "prominence": "central",
                "forecastable": True,
            },
        ],
        "links": [
            {"mention_id": "policy_request", "idea_id": "policy_and_congressional_action", "relation": "expresses"},
            {"mention_id": "local_project_example", "idea_id": "policy_and_congressional_action", "relation": "illustrates"},
        ],
    }
    return record, turns


def test_validates_mentions_ideas_and_many_to_one_links():
    record, turns = _record()
    validated = validate_layered_outcome(record, turns)
    assert validated["validation"]["exact_quote_integrity"] is True
    assert validated["validation"]["mention_count"] == 2
    assert len(idea_targets(validated)) == 1
    assert len(idea_targets(validated, forecastable_only=True)) == 1
    linked = mentions_for_idea(validated, "policy_and_congressional_action")
    assert {item["relation"] for item in linked} == {"expresses", "illustrates"}


def test_quote_token_alignment_repairs_only_unique_exact_word_sequences():
    align = extractor._normalized_token_alignment
    assert align("to respect our great history", "to respect, our great history") == "to respect, our great history"
    assert align("we support this policy", "we support this policy and we support this policy") is None
    assert align("we support that policy", "we support this policy") is None


def test_extractor_records_quote_repairs_and_drops_only_bad_extra_citations():
    result = extractor._map_layered_result(
        {
            "mentions": [{
                "mention_id": "policy_request", "text": "The speaker urges action.",
                "role": "core_expression", "salience": "central",
                "evidence": [
                    {"turn_id": "blind-1", "exact_quote": "invented unrelated quotation"},
                    {"turn_id": "blind-1", "exact_quote": "the speaker supports, action"},
                ],
            }],
            "ideas": [{
                "idea_id": "policy_action", "concept": "Policy action",
                "summary": "The speaker supports action.", "kind": "policy",
                "stance": "support", "prominence": "central", "forecastable": True,
            }],
            "links": [{"mention_id": "policy_request", "idea_id": "policy_action", "relation": "expresses"}],
            "coverage_note": "",
        },
        blind_turns=[{"turn_id": "blind-1", "text": "The speaker supports action."}],
        blind_to_source={"blind-1": "turn-1"},
        metadata={"event_id": "event-1"},
    )
    assert result["mentions"][0]["evidence"] == [{"turn_id": "turn-1", "exact_quote": "The speaker supports action"}]
    assert result["validation"]["normalized_citation_count"] == 1
    assert result["validation"]["rejected_citation_count"] == 1


def test_schema_v2_requires_each_idea_to_have_an_expresses_link():
    record, turns = _record()
    record["schema_version"] = 2
    validated = validate_layered_outcome(record, turns)
    assert validated["validation"]["direct_expression_anchor_validated"] is True

    record["links"][0]["relation"] = "supports"
    with pytest.raises(LayeredOutcomeError, match="direct expression anchor"):
        validate_layered_outcome(record, turns)


def test_schema_v2_expresses_link_is_independent_of_mention_role():
    record, turns = _record()
    record["schema_version"] = 2
    record["mentions"][0]["role"] = "supporting_detail"
    validated = validate_layered_outcome(record, turns)
    assert validated["validation"]["direct_expression_anchor_validated"] is True


def test_forecastable_projection_excludes_event_specific_nonforecastable_ideas():
    record, turns = _record()
    record["ideas"][0]["forecastable"] = False
    validated = validate_layered_outcome(record, turns)
    assert idea_targets(validated, forecastable_only=True) == []
    assert len(idea_targets(validated)) == 1


def test_rejects_missing_link_orphan_idea_and_bad_quote():
    record, turns = _record()
    record["links"].pop()
    with pytest.raises(LayeredOutcomeError, match="unlinked mentions"):
        validate_layered_outcome(record, turns)

    record, turns = _record()
    record["ideas"].append({
        "idea_id": "unlinked_idea", "concept": "Unlinked", "summary": "No transcript support.",
        "kind": "other", "stance": "unclear", "prominence": "brief", "forecastable": False,
    })
    with pytest.raises(LayeredOutcomeError, match="ideas without transcript mentions"):
        validate_layered_outcome(record, turns)

    record, turns = _record()
    record["mentions"][0]["evidence"][0]["exact_quote"] = "a fabricated quote"
    with pytest.raises(LayeredOutcomeError, match="exact transcript substring"):
        validate_layered_outcome(record, turns)


def test_rejects_dangling_and_duplicate_relations():
    record, turns = _record()
    record["links"].append({"mention_id": "missing", "idea_id": "policy_and_congressional_action", "relation": "supports"})
    with pytest.raises(LayeredOutcomeError, match="does not resolve"):
        validate_layered_outcome(record, turns)

    record, turns = _record()
    record["links"].append(dict(record["links"][0]))
    with pytest.raises(LayeredOutcomeError, match="duplicates"):
        validate_layered_outcome(record, turns)
