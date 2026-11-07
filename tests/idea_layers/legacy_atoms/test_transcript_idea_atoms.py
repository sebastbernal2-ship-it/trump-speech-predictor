from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/idea_layers/legacy_atoms/extract_transcript_idea_atoms.py"
SPEC = importlib.util.spec_from_file_location("extract_transcript_idea_atoms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
atoms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atoms)
MERGE_SCRIPT = Path(__file__).parents[3] / "research/idea_layers/legacy_atoms/merge_transcript_idea_atom_retries.py"
MERGE_SPEC = importlib.util.spec_from_file_location("merge_transcript_idea_atom_retries", MERGE_SCRIPT)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
merge_atoms = importlib.util.module_from_spec(MERGE_SPEC)
MERGE_SPEC.loader.exec_module(merge_atoms)


def _valid_response() -> dict:
    return {
        "status": "atoms_found",
        "empty_reason": "",
        "atoms": [{
            "idea_id": "trade_policy",
            "concept": "Trade policy",
            "summary": "The speaker supports changing trade policy.",
            "kind": "policy",
            "stance": "support",
            "prominence": "secondary",
            "confidence": "high",
            "forecastable": True,
            "evidence": [{"turn_id": "blind:turn:0000", "exact_quote": "We need fair trade."}],
        }],
    }


def test_target_turns_include_only_target_speaker_and_keep_raw_text_fallback():
    record = {
        "target_speaker": "speaker",
        "turns": [
            {"turn_id": "t1", "speaker_name": "Speaker", "raw_text": "  We need fair trade.  "},
            {"turn_id": "t2", "speaker_name": "Audience", "text": "Applause."},
            {"turn_id": "t3", "speaker_role": "target_speaker", "text": "And strong jobs."},
        ],
    }
    assert atoms._target_turns(record) == [
        {"turn_id": "t1", "text": "We need fair trade."},
        {"turn_id": "t3", "text": "And strong jobs."},
    ]


def test_target_turn_with_missing_id_is_not_silently_dropped():
    with pytest.raises(ValueError, match="missing a turn_id"):
        atoms._target_turns({"turns": [{"speaker_role": "target_speaker", "text": "A claim."}]})


def test_blind_turn_mapping_removes_source_event_identity():
    blind, mapping = atoms._blind_turns(
        [{"turn_id": "event-123:turn:0008", "text": "We need fair trade."}],
        "atom-case-0001",
    )
    assert blind == [{"turn_id": "atom-case-0001:turn:0000", "text": "We need fair trade."}]
    assert mapping == {"atom-case-0001:turn:0000": "event-123:turn:0008"}
    assert "event-123" not in atoms._prompt(blind)


def test_response_validation_accepts_exact_cited_target_quote():
    response = atoms._validate_response(
        _valid_response(),
        [{"turn_id": "blind:turn:0000", "text": "We need fair trade."}],
    )
    assert response["atoms"][0]["concept"] == "Trade policy"


def test_response_validation_rejects_nonverbatim_or_wrong_turn_evidence():
    source = [{"turn_id": "blind:turn:0000", "text": "We need fair trade."}]
    invalid_quote = _valid_response()
    invalid_quote["atoms"][0]["evidence"][0]["exact_quote"] = "We need fair markets."
    with pytest.raises(ValueError, match="exact transcript substring"):
        atoms._validate_response(invalid_quote, source)
    invalid_turn = _valid_response()
    invalid_turn["atoms"][0]["evidence"][0]["turn_id"] = "someone-else"
    with pytest.raises(ValueError, match="unknown target-speaker turn"):
        atoms._validate_response(invalid_turn, source)


def test_empty_statuses_are_explicit_and_require_a_reason():
    empty = {"status": "no_supported_idea", "empty_reason": "Only greetings.", "atoms": []}
    assert atoms._validate_response(empty, [])["status"] == "no_supported_idea"
    empty["empty_reason"] = ""
    with pytest.raises(ValueError, match="requires an explanation"):
        atoms._validate_response(empty, [])


def test_response_validation_rejects_mixed_empty_status_and_atoms():
    response = _valid_response()
    response["status"] = "no_supported_idea"
    response["empty_reason"] = "No ideas."
    with pytest.raises(ValueError, match="must not include atoms"):
        atoms._validate_response(response, [{"turn_id": "blind:turn:0000", "text": "We need fair trade."}])


def test_json_parser_accepts_fenced_json():
    assert atoms._parse_json_object('```json\n{"status":"atoms_found"}\n```') == {"status": "atoms_found"}


def _write_run(path: Path, row: dict) -> None:
    raw = (atoms.json.dumps(row, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    Path(str(path) + ".manifest.json").write_text(
        atoms.json.dumps({"output_sha256": merge_atoms._sha256(raw), "limitations": []}),
        encoding="utf-8",
    )


def test_retry_merge_replaces_only_matching_failed_event(tmp_path: Path):
    source_fields = {
        "event_id": "e1", "event_date": "2019-01-01",
        "source_record_sha256": "a" * 64, "target_transcript_sha256": "b" * 64,
    }
    base = tmp_path / "base.jsonl"
    retry = tmp_path / "retry.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_run(base, {**source_fields, "extraction_status": "model_error", "atoms": []})
    _write_run(retry, {**source_fields, "extraction_status": "atoms_found", "atoms": [{"forecastable": True}]})
    result = merge_atoms.merge(base, [retry], output)
    row = atoms.json.loads(output.read_text().strip())
    assert result["atoms_found"] == 1
    assert row["extraction_status"] == "atoms_found"
    assert row["retry_provenance"]["replaces_base_output_sha256"] == merge_atoms._sha256(base.read_bytes())


def test_retry_merge_rejects_a_nonmatching_transcript(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    retry = tmp_path / "retry.jsonl"
    _write_run(base, {
        "event_id": "e1", "event_date": "2019-01-01", "source_record_sha256": "a" * 64,
        "target_transcript_sha256": "b" * 64, "extraction_status": "model_error", "atoms": [],
    })
    _write_run(retry, {
        "event_id": "e1", "event_date": "2019-01-01", "source_record_sha256": "a" * 64,
        "target_transcript_sha256": "c" * 64, "extraction_status": "atoms_found", "atoms": [],
    })
    with pytest.raises(ValueError, match="mismatched target_transcript_sha256"):
        merge_atoms.merge(base, [retry], tmp_path / "merged.jsonl")
