from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "research/idea_layers/prediction/generate_open_vocabulary_context_ideas.py"
SPEC = importlib.util.spec_from_file_location("generate_open_vocabulary_context_ideas", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_prompt_is_headline_only_and_requests_open_vocabulary_evidence():
    prompt = module._prompt([
        {"headline_id": "h001", "title": "Congress debates the SAVE America Act"},
    ])
    assert "Do not use tools, outside knowledge, event identity, target transcripts" in module._SYSTEM_PROMPT
    assert "headline_ids" in prompt
    assert "probabilities or confidence scores" in prompt
    assert "SAVE America Act" in prompt
    assert "event_id" not in prompt


def test_candidate_parser_requires_unique_specific_candidates_and_known_evidence():
    parsed = module._parse_candidates(
        json.dumps({"candidates": [{
            "idea": "Voter identification requirements in the SAVE Act",
            "headline_ids": ["h001"],
            "rationale": "The cited headline is explicitly about the proposed act.",
        }]}),
        {"h001"},
    )
    assert parsed[0]["rank"] == 1
    with pytest.raises(ValueError, match="supplied headline IDs"):
        module._parse_candidates(
            json.dumps({"candidates": [{"idea": "Border enforcement", "headline_ids": ["h999"], "rationale": "headline"}]}),
            {"h001"},
        )
    with pytest.raises(ValueError, match="unique"):
        module._parse_candidates(
            json.dumps({"candidates": [
                {"idea": "Border enforcement", "headline_ids": ["h001"], "rationale": "a"},
                {"idea": " border  enforcement ", "headline_ids": ["h001"], "rationale": "b"},
            ]}),
            {"h001"},
        )


def test_runner_generates_without_transcript_prompt_and_rejects_same_day_context(tmp_path, monkeypatch):
    event = {
        "event_id": "evt", "event_date": "2025-10-29", "target_speaker": "Speaker",
        "turns": [{"turn_id": "t1", "speaker_name": "Speaker", "speaker_role": "target_speaker", "text": "We need voter ID."}],
    }
    snapshot = {
        "event_id": "evt", "event_date": "2025-10-29", "exploratory_only": True,
        "candidate_sources": [
            {"title": "Congress debates the SAVE America Act", "published_at": "2025-10-28", "source_id": "src1"},
            {"title": "Same day leak", "published_at": "2025-10-29", "source_id": "src2"},
        ],
    }
    events_path, snapshots_path = tmp_path / "events.jsonl", tmp_path / "snapshots.jsonl"
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    snapshots_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    prompts = []

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"candidates": [{
            "idea": "Voter identification requirements in the SAVE Act",
            "headline_ids": ["h001"],
            "rationale": "The title names the legislation.",
        }]})

    monkeypatch.setattr(module, "_call_pi", fake_call)
    output = tmp_path / "predictions.jsonl"
    result = module.run(
        events_path=events_path,
        snapshots_path=snapshots_path,
        output_path=output,
        event_ids={"evt"},
        provider="openai-codex",
        model="gpt-6.1-sol",
    )
    row = json.loads(output.read_text().splitlines()[0])
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert result["successes"] == 1
    assert row["eligible_headline_count"] == 1
    assert row["headline_evidence"]["h001"]["source_id"] == "src1"
    assert row["weak_target_ids_posthoc_only"]
    assert "We need voter identification" not in prompts[0]
    assert "voter id" not in prompts[0].casefold()
    assert manifest["target_transcripts_used_in_prompt"] is False
    assert manifest["transcript_labels_used_only_posthoc_for_scoring"] is True
