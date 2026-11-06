from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "research/merge_layered_transcript_outcome_retries.py"
SPEC = importlib.util.spec_from_file_location("merge_layered_transcript_outcome_retries", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
merge_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_module)


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return raw


def _event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "target_speaker": "speaker",
        "turns": [{"turn_id": f"{event_id}:t1", "speaker_role": "target_speaker", "text": "The speaker supports this policy."}],
    }


def _success_row(event: dict) -> dict:
    turn = event["turns"][0]
    transcript_hash = merge_module._sha256(f"{turn['turn_id']}\t{turn['text']}".encode())
    return {
        "schema_version": 1,
        "artifact_type": "layered-transcript-outcome",
        "status": "validated_candidate",
        "event_id": event["event_id"],
        "source_record_sha256": merge_module._canonical_hash(event),
        "target_transcript_sha256": transcript_hash,
        "mentions": [{
            "mention_id": "policy_support",
            "text": "The speaker supports the policy.",
            "role": "core_expression",
            "salience": "central",
            "evidence": [{"turn_id": turn["turn_id"], "exact_quote": "supports this policy"}],
        }],
        "ideas": [{
            "idea_id": "policy_idea", "concept": "Policy", "summary": "The speaker supports a policy.",
            "kind": "policy", "stance": "support", "prominence": "central", "forecastable": True,
        }],
        "links": [{"mention_id": "policy_support", "idea_id": "policy_idea", "relation": "expresses"}],
    }


def test_merge_replaces_only_failed_rows_and_revalidates_retry_quotes(tmp_path: Path):
    event = _event("event-1")
    event_path = tmp_path / "events.jsonl"
    event_bytes = _write_jsonl(event_path, [event])
    base_path = tmp_path / "base.jsonl"
    base_row = {"event_id": "event-1", "source_record_sha256": merge_module._canonical_hash(event), "status": "model_error"}
    base_bytes = _write_jsonl(base_path, [base_row])
    Path(str(base_path) + ".manifest.json").write_text(json.dumps({
        "events_sha256": merge_module._sha256(event_bytes),
        "output_sha256": merge_module._sha256(base_bytes),
    }))
    retry_path = tmp_path / "retry.jsonl"
    retry_bytes = _write_jsonl(retry_path, [_success_row(event)])
    Path(str(retry_path) + ".manifest.json").write_text(json.dumps({
        "events_sha256": merge_module._sha256(event_bytes),
        "output_sha256": merge_module._sha256(retry_bytes),
    }))

    output_path = tmp_path / "merged.jsonl"
    result = merge_module.merge(events_path=event_path, base_path=base_path, retry_paths=[retry_path], output_path=output_path)
    merged = json.loads(output_path.read_text().splitlines()[0])
    assert result["replaced_errors"] == 1
    assert result["unresolved_errors"] == 0
    assert merged["status"] == "validated_candidate"
    assert merged["validation"]["exact_quote_integrity"] is True


def test_merge_preserves_unresolved_retry_failure(tmp_path: Path):
    event = _event("event-1")
    event_path = tmp_path / "events.jsonl"
    event_bytes = _write_jsonl(event_path, [event])
    base_path = tmp_path / "base.jsonl"
    base_row = {"event_id": "event-1", "source_record_sha256": merge_module._canonical_hash(event), "status": "model_error"}
    base_bytes = _write_jsonl(base_path, [base_row])
    Path(str(base_path) + ".manifest.json").write_text(json.dumps({
        "events_sha256": merge_module._sha256(event_bytes), "output_sha256": merge_module._sha256(base_bytes),
    }))
    retry_path = tmp_path / "retry.jsonl"
    retry_bytes = _write_jsonl(retry_path, [{"event_id": "event-1", "status": "model_error"}])
    Path(str(retry_path) + ".manifest.json").write_text(json.dumps({
        "events_sha256": merge_module._sha256(event_bytes), "output_sha256": merge_module._sha256(retry_bytes),
    }))
    result = merge_module.merge(events_path=event_path, base_path=base_path, retry_paths=[retry_path], output_path=tmp_path / "merged.jsonl")
    assert result["replaced_errors"] == 0
    assert result["unresolved_errors"] == 1
