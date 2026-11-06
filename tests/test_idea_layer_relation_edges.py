from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from speechprefix.idea_layer.semantic_anchors import (
    build_expression_anchor_record,
    validate_expression_anchor_record,
)
from speechprefix.idea_layer.relation_edges import (
    FINE_RELATIONS,
    build_relation_candidate_record,
    candidate_edge_id,
    direct_expression_anchors,
    validate_relation_candidate_record,
)


def _fixture():
    turns = {
        "t1": "The policy should lower costs.",
        "t2": "For example, the current fee adds hundreds of dollars.",
    }
    outcome = {
        "schema_version": 2,
        "artifact_type": "layered-transcript-outcome",
        "event_id": "event_a",
        "source_record_sha256": "a" * 64,
        "target_transcript_sha256": "b" * 64,
        "mentions": [
            {"mention_id": "policy_claim", "text": "The policy lowers costs.", "role": "core_expression", "salience": "central",
             "evidence": [{"turn_id": "t1", "exact_quote": "The policy should lower costs."}]},
            {"mention_id": "fee_example", "text": "A fee adds hundreds of dollars.", "role": "illustrative_example", "salience": "secondary",
             "evidence": [{"turn_id": "t2", "exact_quote": "the current fee adds hundreds of dollars"}]},
        ],
        "ideas": [{"idea_id": "lower_costs", "concept": "Lowering costs", "summary": "The policy should lower costs.",
                   "kind": "policy", "stance": "support", "prominence": "central", "forecastable": True}],
        "links": [
            {"mention_id": "policy_claim", "idea_id": "lower_costs", "relation": "expresses"},
            {"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates"},
        ],
    }
    return outcome, turns


def _anchor_record(outcome, turns, *, unanchored=None):
    anchors = [] if unanchored else [{"mention_id": "policy_claim", "idea_id": "lower_costs"}]
    return build_expression_anchor_record(
        outcome, anchors, unanchored or [], provider="anchor-provider", model="anchor-model",
        engine_version="anchor-v1", prompt_version="anchor-p1", turns_by_id=turns,
    )


def test_anchor_pass_separates_direct_expression_and_explicit_abstention():
    outcome, turns = _fixture()
    record = build_expression_anchor_record(
        outcome,
        [{"mention_id": "policy_claim", "idea_id": "lower_costs"}],
        [],
        provider="test-provider", model="test-model", engine_version="anchor-v1", prompt_version="anchor-p1",
        turns_by_id=turns,
    )
    assert record["anchors"][0]["mention_id"] == "policy_claim"
    assert record["unanchored_idea_ids"] == []
    assert record["validation"]["anchor_completeness_validated"] is False
    assert validate_expression_anchor_record(record, outcome, turns) is record


def test_anchor_pass_can_abstain_without_inventing_a_direct_mention():
    outcome, turns = _fixture()
    record = _anchor_record(outcome, turns, unanchored=["lower_costs"])
    assert record["anchors"] == []
    assert record["unanchored_idea_ids"] == ["lower_costs"]
    tampered = copy.deepcopy(record)
    tampered["unanchored_idea_ids"] = []
    with pytest.raises(ValueError, match="each source idea"):
        validate_expression_anchor_record(tampered, outcome, turns)


def test_anchor_artifact_requires_exact_partition_of_source_ideas():
    outcome, turns = _fixture()
    with pytest.raises(ValueError, match="each source idea"):
        build_expression_anchor_record(
            outcome, [], [], provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
        )
    with pytest.raises(ValueError, match="must contain exactly"):
        build_expression_anchor_record(
            outcome, [{"mention_id": "policy_claim", "idea_id": "lower_costs", "relation": "expresses"}], [],
            provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
        )


def test_relation_artifact_binds_separate_anchors_and_keeps_fine_edges_separate():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    assert direct_expression_anchors(anchors) == [{"mention_id": "policy_claim", "idea_id": "lower_costs"}]
    record = build_relation_candidate_record(
        outcome, anchors,
        [{"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates"}],
        provider="test-provider", model="test-model", engine_version="edge-v1", prompt_version="prompt-v1",
        turns_by_id=turns,
    )
    assert record["direct_expression_anchors"] == direct_expression_anchors(anchors)
    assert record["edges"] == [{
        "edge_id": candidate_edge_id("fee_example", "lower_costs", "illustrates"),
        "mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates",
    }]
    assert "expresses" not in {edge["relation"] for edge in record["edges"]}
    assert record["validation"]["semantic_relation_validated"] is False
    assert validate_relation_candidate_record(record, outcome, anchors, turns) is record


def test_empty_relation_candidate_set_is_valid_and_not_claimed_complete():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    record = build_relation_candidate_record(
        outcome, anchors, [], provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
    )
    assert record["edges"] == []
    assert record["validation"]["edge_completeness_validated"] is False


@pytest.mark.parametrize("edge", [
    {"mention_id": "policy_claim", "idea_id": "lower_costs", "relation": "supports"},
    {"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "expresses"},
    {"mention_id": "unknown", "idea_id": "lower_costs", "relation": "illustrates"},
    {"mention_id": "fee_example", "idea_id": "unknown", "relation": "illustrates"},
    {"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "supports", "extra": True},
])
def test_rejects_anchor_duplicates_bad_nodes_forced_expresses_and_unknown_fields(edge):
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    with pytest.raises(ValueError):
        build_relation_candidate_record(
            outcome, anchors, [edge], provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
        )


def test_relation_edges_cannot_point_to_unanchored_ideas():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns, unanchored=["lower_costs"])
    with pytest.raises(ValueError, match="anchored idea"):
        build_relation_candidate_record(
            outcome, anchors,
            [{"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates"}],
            provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
        )


def test_rejects_duplicate_or_multiple_relation_labels_for_a_pair():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    edges = [
        {"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates"},
        {"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "supports"},
    ]
    with pytest.raises(ValueError, match="multiple fine relations"):
        build_relation_candidate_record(
            outcome, anchors, edges, provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
        )


def test_rejects_tampered_anchor_projection_and_source_outcome_hash():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    record = build_relation_candidate_record(
        outcome, anchors, [], provider="test", model="test", engine_version="v1", prompt_version="p1", turns_by_id=turns,
    )
    tampered = copy.deepcopy(record)
    tampered["direct_expression_anchors"] = []
    with pytest.raises(ValueError, match="exactly project"):
        validate_relation_candidate_record(tampered, outcome, anchors, turns)
    tampered = copy.deepcopy(record)
    tampered["source_outcome_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source outcome hash"):
        validate_relation_candidate_record(tampered, outcome, anchors, turns)


def test_relationship_audit_requires_full_anchor_and_edge_reviews():
    audit_path = Path(__file__).parents[1] / "research/audit_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("audit_layered_relation_edges", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = {
        "anchor_reviews": [{"anchor_id": "anchor_a", "verdict": "direct", "note": "The mention states the idea."}],
        "missing_anchor_candidates": [{"mention_id": "m2", "idea_id": "i1", "note": "This mention also states the idea."}],
        "edge_reviews": [{"edge_id": "edge_a", "verdict": "plausible", "note": "The detail illustrates the idea."}],
    }
    result = module._validate_response(
        response,
        expected_anchor_ids={"anchor_a"},
        expected_edge_ids={"edge_a"},
        mention_ids={"m1", "m2"},
        idea_ids={"i1"},
        existing_anchor_pairs={("m1", "i1")},
    )
    assert result["anchor_reviews"][0]["verdict"] == "direct"
    assert result["edge_reviews"][0]["edge_id"] == "edge_a"
    assert result["missing_anchor_candidates"] == response["missing_anchor_candidates"]
    response["edge_reviews"] = []
    with pytest.raises(ValueError, match="omitted"):
        module._validate_response(
            response,
            expected_anchor_ids={"anchor_a"},
            expected_edge_ids={"edge_a"},
            mention_ids={"m1", "m2"},
            idea_ids={"i1"},
            existing_anchor_pairs={("m1", "i1")},
        )
    response["edge_reviews"] = [{"edge_id": "edge_a", "verdict": "plausible", "note": "ok"}]
    response["missing_anchor_candidates"] = [{"mention_id": "m1", "idea_id": "i1", "note": "duplicate anchor"}]
    with pytest.raises(ValueError, match="duplicates"):
        module._validate_response(
            response,
            expected_anchor_ids={"anchor_a"},
            expected_edge_ids={"edge_a"},
            mention_ids={"m1", "m2"},
            idea_ids={"i1"},
            existing_anchor_pairs={("m1", "i1")},
        )


def test_relation_prompt_only_receives_frozen_anchors_not_legacy_fine_links():
    script_path = Path(__file__).parents[1] / "research/extract_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("extract_layered_relation_edges", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    prompt = module._prompt(
        [{"turn_id": "blind:turn:0000", "text": "The policy should lower costs."}],
        {"mentions": [{"mention_id": "policy_claim", "text": "lower costs"}],
         "ideas": [{"idea_id": "lower_costs", "forecastable": True}],
         "links": [{"mention_id": "policy_claim", "idea_id": "lower_costs", "relation": "legacy_relation_must_not_be_seen"}]},
        anchors,
        [{"mention_id": "policy_claim", "idea_id": "lower_costs"}],
    )
    assert "legacy_relation_must_not_be_seen" not in prompt
    assert '"direct_expression_anchors"' in prompt
    assert '"forecastable"' not in prompt
    assert set(module._parse_edges('{"edges": [{"mention_id":"m","idea_id":"i","relation":"supports"}]}')[0]) == {"mention_id", "idea_id", "relation"}
    assert FINE_RELATIONS == {"supports", "illustrates", "elaborates", "qualifies", "contextualizes"}
