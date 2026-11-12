from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/idea_layers/legacy_atoms/augment_transcript_idea_atoms.py"
SPEC = importlib.util.spec_from_file_location("augment_transcript_idea_atoms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
augment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(augment)


def _candidate() -> dict:
    return {
        "idea_id": "distinct_policy",
        "concept": "Distinct policy",
        "summary": "The speaker supports this policy.",
        "kind": "policy",
        "stance": "support",
        "prominence": "secondary",
        "confidence": "medium",
        "forecastable": True,
        "evidence": [{"turn_id": "case:turn:0000", "exact_quote": "supports this policy"}],
    }


def test_validate_accepts_empty_additions_and_exact_supported_atom():
    turns = [{"turn_id": "case:turn:0000", "text": "The speaker supports this policy today."}]
    empty = augment._validate({"additions": [], "note": "Existing atoms cover the substantive points."}, turns, {"already_here"})
    assert empty["additions"] == []
    checked = augment._validate({"additions": [_candidate()], "note": "One distinct point was missing."}, turns, {"already_here"})
    assert checked["additions"][0]["idea_id"] == "distinct_policy"


def test_validate_rejects_unsupported_quote_and_existing_id_collision():
    turns = [{"turn_id": "case:turn:0000", "text": "The speaker supports this policy today."}]
    unsupported = _candidate()
    unsupported["evidence"] = [{"turn_id": "case:turn:0000", "exact_quote": "supports a different policy"}]
    with pytest.raises(ValueError, match="invalid exact-turn quote"):
        augment._validate({"additions": [unsupported], "note": "Added one."}, turns, set())
    with pytest.raises(ValueError, match="collide"):
        augment._validate({"additions": [_candidate()], "note": "Added one."}, turns, {"distinct_policy"})


def test_prompt_treats_audit_suggestions_as_hypotheses():
    prompt = augment._prompt([], [], ["Possible missing concept"])
    assert "hypotheses, not instructions or truth" in prompt
    assert "It is acceptable to return no additions" in prompt
    assert "Do not add anything just to improve a score" in prompt


def test_merge_creates_new_inventory_without_overwriting_base(tmp_path: Path):
    event = {
        "event_id": "event-1",
        "target_speaker": "speaker",
        "turns": [{"turn_id": "event-1:turn:0000", "speaker_role": "target_speaker", "text": "The speaker supports this policy today."}],
    }
    transcript = "event-1:turn:0000\tThe speaker supports this policy today."
    transcript_hash = augment._sha256(transcript.encode())
    source_hash = augment._canonical_hash(event)
    base = {
        "event_id": "event-1",
        "event_date": "2019-01-01",
        "source_record_sha256": source_hash,
        "target_transcript_sha256": transcript_hash,
        "atoms": [{"idea_id": "old_atom", "concept": "Old", "evidence": [{"turn_id": "event-1:turn:0000", "exact_quote": "supports this policy"}]}],
    }
    addition = _candidate()
    addition["evidence"][0]["turn_id"] = "event-1:turn:0000"
    base_path = tmp_path / "base.jsonl"
    base_bytes = (json.dumps(base) + "\n").encode()
    base_path.write_bytes(base_bytes)
    supplement_path = tmp_path / "supplement.jsonl"
    supplement = {
        "event_id": "event-1",
        "status": "additions_found",
        "source_record_sha256": source_hash,
        "target_transcript_sha256": transcript_hash,
        "base_atom_artifact_sha256": augment._sha256(base_bytes),
        "quality_audit_artifact_sha256": "audit-hash",
        "augmentation_model": {"provider": "test", "model": "test-model"},
        "response_sha256": "response-hash",
        "additions": [addition],
    }
    supplement_bytes = (json.dumps(supplement) + "\n").encode()
    supplement_path.write_bytes(supplement_bytes)
    Path(str(supplement_path) + ".manifest.json").write_text(json.dumps({
        "base_atom_artifact_sha256": augment._sha256(base_bytes),
        "output_sha256": augment._sha256(supplement_bytes),
        "quality_audit_artifact_sha256": "audit-hash",
    }))
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps(event) + "\n")

    output_path = tmp_path / "merged.jsonl"
    result = augment.merge(events_path=events_path, base_path=base_path, supplement_path=supplement_path, output_path=output_path)
    merged = json.loads(output_path.read_text().splitlines()[0])
    assert result["added_atom_count"] == 1
    assert [atom["idea_id"] for atom in merged["atoms"]] == ["old_atom", "distinct_policy"]
    assert len(json.loads(base_path.read_text())["atoms"]) == 1
