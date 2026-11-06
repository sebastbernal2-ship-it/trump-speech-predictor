from __future__ import annotations

import pytest

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
