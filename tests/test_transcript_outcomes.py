from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "research/extract_layered_transcript_outcomes.py"
SPEC = importlib.util.spec_from_file_location("extract_layered_transcript_outcomes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)
CHUNKED_SCRIPT = Path(__file__).parents[1] / "research/extract_layered_transcript_outcomes_chunked.py"
CHUNKED_SPEC = importlib.util.spec_from_file_location("extract_layered_transcript_outcomes_chunked", CHUNKED_SCRIPT)
assert CHUNKED_SPEC is not None and CHUNKED_SPEC.loader is not None
chunked_extractor = importlib.util.module_from_spec(CHUNKED_SPEC)
CHUNKED_SPEC.loader.exec_module(chunked_extractor)
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


def test_extractor_records_safe_fallbacks_for_unknown_enum_values():
    result = extractor._map_layered_result(
        {
            "mentions": [{
                "mention_id": "m1", "text": "A transcript detail.", "role": "unsupported_role",
                "salience": "unsupported_salience",
                "evidence": [{"turn_id": "blind-1", "exact_quote": "the speaker supports action"}],
            }],
            "ideas": [{
                "idea_id": "i1", "concept": "Action", "summary": "The speaker supports action.",
                "kind": "unsupported_kind", "stance": "neutral", "prominence": "high", "forecastable": False,
            }],
            "links": [{"mention_id": "m1", "idea_id": "i1", "relation": "expresses"}],
            "coverage_note": "",
        },
        blind_turns=[{"turn_id": "blind-1", "text": "the speaker supports action"}],
        blind_to_source={"blind-1": "turn-1"},
        metadata={"event_id": "event-1"},
    )
    assert result["mentions"][0]["role"] == "other"
    assert result["mentions"][0]["salience"] == "secondary"
    assert result["ideas"][0]["kind"] == "other"
    assert result["ideas"][0]["stance"] == "unclear"
    assert result["ideas"][0]["prominence"] == "secondary"
    assert result["validation"]["enum_normalization_count"] == 5


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


def test_chunk_partition_preserves_order_and_exact_source_slices():
    first = " ".join(f"word{i}" for i in range(202))
    second = " ".join(f"tail{i}" for i in range(70))
    chunks = chunked_extractor._partition_turns(
        [{"turn_id": "turn-1", "text": first}, {"turn_id": "turn-2", "text": second}],
        100,
    )
    segments = [segment for chunk in chunks for segment in chunk["segments"]]
    assert [token for segment in segments for token in segment["text"].split()] == (first + " " + second).split()
    assert all(chunk["word_count"] <= 100 for chunk in chunks)
    assert [segment["source_turn_id"] for segment in segments] == ["turn-1"] * 3 + ["turn-2"]


def test_chunk_idea_consolidation_maps_each_candidate_once_and_deduplicates_links():
    candidates = [
        {"idea_id": "chunk_000_a", "stance": "support"},
        {"idea_id": "chunk_001_b", "stance": "support"},
        {"idea_id": "chunk_002_c", "stance": "oppose"},
    ]
    mentions = [{"mention_id": "chunk_000_m", "text": "Direct statement", "role": "core_expression"}]
    links = [
        {"mention_id": "chunk_000_m", "idea_id": "chunk_000_a", "relation": "expresses"},
        {"mention_id": "chunk_000_m", "idea_id": "chunk_001_b", "relation": "expresses"},
        {"mention_id": "chunk_000_m", "idea_id": "chunk_002_c", "relation": "expresses"},
    ]
    groups = {"groups": [
        {"member_idea_ids": ["chunk_000_a", "chunk_001_b"], "concept": "Shared", "summary": "A shared claim.", "kind": "policy", "stance": "support", "prominence": "repeated", "forecastable": True},
        {"member_idea_ids": ["chunk_002_c"], "concept": "Distinct", "summary": "A distinct claim.", "kind": "political", "stance": "oppose", "prominence": "secondary", "forecastable": False},
    ]}
    ideas, final_links, provenance = chunked_extractor._consolidate(groups, candidates, mentions, links)
    assert len(ideas) == 2
    assert len(final_links) == 2
    assert provenance[0]["member_idea_ids"] == ["chunk_000_a", "chunk_001_b"]


def test_incremental_chunk_merge_preserves_provenance_and_remaps_links(monkeypatch):
    def idea(idea_id):
        return {
            "idea_id": idea_id, "concept": "Trade policy", "summary": "The speaker supports the trade agreement.",
            "kind": "policy", "stance": "support", "prominence": "central", "forecastable": True,
        }

    bundles = []
    for chunk_id, idea_id, mention_id in (
        ("chunk_000", "chunk_000_trade", "chunk_000_m"),
        ("chunk_001", "chunk_001_trade", "chunk_001_m"),
    ):
        bundles.append({
            "chunk_id": chunk_id,
            "ideas": [idea(idea_id)],
            "mentions": [{"mention_id": mention_id, "text": "A direct trade statement.", "role": "core_expression"}],
            "links": [{"mention_id": mention_id, "idea_id": idea_id, "relation": "expresses"}],
        })

    def fake_call(prompt, *, provider, model, timeout, validate, label, max_attempts=5):
        candidate_ids = ["chunk_000_trade", "chunk_001_trade"]
        response = {"groups": [{
            "member_idea_ids": candidate_ids, "concept": "Trade policy",
            "summary": "The speaker supports the trade agreement.", "kind": "policy",
            "stance": "support", "prominence": "repeated", "forecastable": True,
        }]}
        return validate(response), "fake-hash", 1

    monkeypatch.setattr(chunked_extractor, "_call_with_repairs", fake_call)
    ideas, links, provenance, steps, attempts = chunked_extractor._incremental_consolidate(
        bundles, provider="test", model="test", timeout=1, event_id="event-1",
    )
    assert len(ideas) == 1
    assert ideas[0]["idea_id"] == "chunk_000_trade"
    assert provenance == [{"idea_id": "chunk_000_trade", "member_idea_ids": ["chunk_000_trade", "chunk_001_trade"]}]
    assert {link["mention_id"] for link in links} == {"chunk_000_m", "chunk_001_m"}
    assert all(link["idea_id"] == "chunk_000_trade" for link in links)
    assert len(steps) == 1 and attempts == 1


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
