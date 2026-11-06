from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from speechprefix.idea_layer.semantic_anchors import (
    build_expression_anchor_record,
    validate_expression_anchor_record,
)
from speechprefix.idea_layer.relation_edges import (
    FINE_RELATION_BOUNDARY_EXAMPLES,
    FINE_RELATION_DECISION_RULES,
    FINE_RELATION_DEFINITIONS,
    FINE_RELATION_TAXONOMY_VERSION,
    FINE_RELATIONS,
    build_relation_candidate_record,
    build_relation_error_record,
    candidate_edge_id,
    canonical_sha256,
    direct_expression_anchors,
    validate_proposed_relation_edges,
    validate_relation_candidate_record,
    validate_relation_error_record,
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
    assert record["extractor"]["taxonomy_version"] == FINE_RELATION_TAXONOMY_VERSION
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
    assert '"allowed_mention_ids"' in prompt
    assert '"allowed_anchored_idea_ids"' in prompt
    assert '"anchor_pairs_not_to_repeat"' in prompt
    assert FINE_RELATION_TAXONOMY_VERSION in prompt
    assert "A nurse leaving the hospital at 10:30" in prompt
    assert set(module._parse_edges('{"edges": [{"mention_id":"m","idea_id":"i","relation":"supports"}]}')[0]) == {"mention_id", "idea_id", "relation"}
    assert FINE_RELATIONS == {"supports", "illustrates", "elaborates", "qualifies", "contextualizes"}


def test_fine_relation_taxonomy_has_one_generic_boundary_example_per_type():
    assert set(FINE_RELATION_DEFINITIONS) == FINE_RELATIONS
    assert {example["relation"] for example in FINE_RELATION_BOUNDARY_EXAMPLES} == FINE_RELATIONS
    assert any("If multiple labels remain plausible" in rule for rule in FINE_RELATION_DECISION_RULES)
    assert all(example["idea"] == "Extend evening bus service." for example in FINE_RELATION_BOUNDARY_EXAMPLES)


def test_relation_auditor_uses_the_same_versioned_taxonomy():
    audit_path = Path(__file__).parents[1] / "research/audit_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("audit_layered_relation_edges_taxonomy_test", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    prompt = module._prompt(
        [{"turn_id": "blind:t1", "text": turns["t1"]}, {"turn_id": "blind:t2", "text": turns["t2"]}],
        outcome,
        anchors,
        {"edges": []},
        {"t1": "blind:t1", "t2": "blind:t2"},
    )
    assert FINE_RELATION_TAXONOMY_VERSION in prompt
    assert "The last bus leaves at 9:45" in prompt
    assert "A nurse leaving the hospital at 10:30" in prompt


@pytest.mark.parametrize(("edge", "message"), [
    ({"mention_id": "not-in-source", "idea_id": "lower_costs", "relation": "supports"}, "mention_id does not resolve"),
    ({"mention_id": "policy_claim", "idea_id": "lower_costs", "relation": "supports"}, "duplicates an existing direct-expression anchor"),
])
def test_preoutput_relation_validator_rejects_unknown_mentions_and_anchor_duplicates(edge, message):
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    with pytest.raises(ValueError, match=message):
        validate_proposed_relation_edges([edge], outcome, anchors)


def _runner_input_files(tmp_path):
    outcome, turns = _fixture()
    event = {
        "event_id": outcome["event_id"],
        "event_date": "2020-01-01",
        "target_speaker": "speaker",
        "turns": [
            {"turn_id": turn_id, "speaker_name": "speaker", "speaker_role": "target_speaker", "text": text}
            for turn_id, text in turns.items()
        ],
    }
    outcome["source_record_sha256"] = canonical_sha256(event)
    transcript = "\n".join(f"{turn_id}\t{text}" for turn_id, text in turns.items())
    outcome["target_transcript_sha256"] = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    outcome["status"] = "validated_candidate"
    anchors = _anchor_record(outcome, turns)
    paths = {
        "events": tmp_path / "events.jsonl",
        "outcomes": tmp_path / "outcomes.jsonl",
        "anchors": tmp_path / "anchors.jsonl",
    }
    for key, row in (("events", event), ("outcomes", outcome), ("anchors", anchors)):
        paths[key].write_text(json.dumps(row) + "\n", encoding="utf-8")
    return event, outcome, anchors, turns, paths


def test_relation_runner_persists_exhausted_invalid_response_as_explicit_failure(tmp_path, monkeypatch):
    script_path = Path(__file__).parents[1] / "research/extract_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("extract_layered_relation_edges_failure_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, anchors, _, paths = _runner_input_files(tmp_path)
    calls = []
    monkeypatch.setattr(
        module,
        "_call_pi",
        lambda *args, **kwargs: calls.append(1) or json.dumps({
            "edges": [{"mention_id": "not-in-source", "idea_id": "lower_costs", "relation": "supports"}]
        }),
    )
    output = tmp_path / "relation-rows.jsonl"
    result = module.run(
        events_path=paths["events"],
        outcomes_path=paths["outcomes"],
        anchors_path=paths["anchors"],
        output_path=output,
        event_ids={"event_a"},
        provider="test-provider",
        model="test-model",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert len(calls) == 3
    assert result["events"] == manifest["output_row_count"] == 1
    assert result["successful_events"] == manifest["successful_event_count"] == 0
    assert result["model_errors"] == manifest["model_error_count"] == 1
    assert rows[0]["artifact_type"] == "layered-idea-relation-candidate-error"
    assert rows[0]["status"] == "model_error"
    assert rows[0]["attempt_count"] == 3
    assert "mention_id does not resolve" in rows[0]["errors"][0]
    assert "edges" not in rows[0]

    audit_path = Path(__file__).parents[1] / "research/audit_layered_relation_edges.py"
    audit_spec = importlib.util.spec_from_file_location("audit_relation_error_row_test", audit_path)
    assert audit_spec is not None and audit_spec.loader is not None
    audit_module = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit_module)
    anchor_id = anchors["anchors"][0]["anchor_id"]
    monkeypatch.setattr(
        audit_module,
        "_call_pi",
        lambda *args, **kwargs: json.dumps({
            "anchor_reviews": [{"anchor_id": anchor_id, "verdict": "direct", "note": "The mention states the idea."}],
            "missing_anchor_candidates": [],
            "edge_reviews": [],
        }),
    )
    audit_output = tmp_path / "audit.jsonl"
    audit_module.run(
        events_path=paths["events"],
        outcomes_path=paths["outcomes"],
        relations_path=output,
        anchors_path=paths["anchors"],
        output_path=audit_output,
        provider="audit-provider",
        model="audit-model",
    )
    audit_row = json.loads(audit_output.read_text().splitlines()[0])
    assert audit_row["relation_candidate_status"] == "model_error"
    assert audit_row["fine_edge_review_status"] == "not_run_source_extraction_failed"
    assert audit_row["edge_count"] is None
    assert audit_row["edge_verdict_counts"] is None
    assert audit_row["edge_reviews"] is None
    assert audit_row["anchor_verdict_counts"]["direct"] == 1


def test_relation_runner_records_repair_attempts_on_successful_candidate(tmp_path, monkeypatch):
    script_path = Path(__file__).parents[1] / "research/extract_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("extract_layered_relation_edges_repair_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, _, _, paths = _runner_input_files(tmp_path)
    responses = iter([
        json.dumps({"edges": [{"mention_id": "not-in-source", "idea_id": "lower_costs", "relation": "supports"}]}),
        json.dumps({"edges": [{"mention_id": "fee_example", "idea_id": "lower_costs", "relation": "illustrates"}]}),
    ])
    monkeypatch.setattr(module, "_call_pi", lambda *args, **kwargs: next(responses))
    output = tmp_path / "relation-repaired.jsonl"
    result = module.run(
        events_path=paths["events"],
        outcomes_path=paths["outcomes"],
        anchors_path=paths["anchors"],
        output_path=output,
        event_ids={"event_a"},
        provider="test-provider",
        model="test-model",
    )
    row = json.loads(output.read_text().splitlines()[0])
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert result["successful_events"] == manifest["successful_event_count"] == 1
    assert result["repaired_events"] == manifest["repaired_event_count"] == 1
    assert result["repair_attempts"] == manifest["repair_attempt_count"] == 1
    assert row["extractor"]["attempt_count"] == 2
    assert "mention_id does not resolve" in row["extractor"]["repair_errors"][0]
    assert row["edges"][0]["relation"] == "illustrates"


def test_relation_error_record_is_bound_and_audit_prompt_does_not_call_it_zero_edges():
    outcome, turns = _fixture()
    anchors = _anchor_record(outcome, turns)
    error = build_relation_error_record(
        outcome,
        anchors,
        ["ValueError: invalid mention ID"],
        attempt_count=1,
        provider="test-provider",
        model="test-model",
        engine_version="edge-v1",
        prompt_version="prompt-v1",
        turns_by_id=turns,
    )
    assert error["status"] == "model_error"
    assert validate_relation_error_record(error, outcome, anchors, turns) is error
    audit_path = Path(__file__).parents[1] / "research/audit_layered_relation_edges.py"
    spec = importlib.util.spec_from_file_location("audit_layered_relation_edges_error_test", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = module._prompt(
        [{"turn_id": "blind:t1", "text": turns["t1"]}, {"turn_id": "blind:t2", "text": turns["t2"]}],
        outcome,
        anchors,
        error,
        {"t1": "blind:t1", "t2": "blind:t2"},
    )
    assert "relation extraction failed" in prompt
    assert '"candidate_fine_edges": []' in prompt
    tampered = copy.deepcopy(error)
    tampered["source_anchor_artifact_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source anchor artifact hash"):
        validate_relation_error_record(tampered, outcome, anchors, turns)
