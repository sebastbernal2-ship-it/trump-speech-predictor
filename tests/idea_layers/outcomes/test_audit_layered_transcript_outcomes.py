from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/idea_layers/outcomes/audit_layered_transcript_outcomes.py"
SPEC = importlib.util.spec_from_file_location("audit_layered_transcript_outcomes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_permanent_provider_rejections_are_detected_without_retaining_error_payloads():
    assert audit._is_permanent_provider_rejection('402: {"message":"Insufficient Balance"}')
    assert audit._is_permanent_provider_rejection("401 Unauthorized")
    assert not audit._is_permanent_provider_rejection("temporary network timeout")


def test_sample_selection_spreads_by_length_and_excludes_same_provider():
    rows = [
        {
            "event_id": f"event-{index}",
            "status": "validated_candidate",
            "target_speaker_word_count": index * 100,
            "extractor": {"provider": "deepseek" if index == 5 else "openai-codex"},
        }
        for index in range(6)
    ]
    selected, excluded, same_model = audit._select_sample(
        rows,
        events={row["event_id"]: {} for row in rows},
        auditor_provider="deepseek",
        auditor_model="deepseek-v4-pro",
        sample_size=2,
        include_event_ids={"event-4"},
    )
    assert [row["event_id"] for row in selected] == ["event-0", "event-3", "event-4"]
    assert excluded == ["event-5"]
    assert same_model == []


def test_same_provider_different_model_requires_opt_in_and_never_self_audits():
    rows = [
        {"event_id": "different-model", "status": "validated_candidate", "target_speaker_word_count": 1,
         "extractor": {"provider": "openai-codex", "model": "gpt-6-luna"}},
        {"event_id": "same-model", "status": "validated_candidate", "target_speaker_word_count": 2,
         "extractor": {"provider": "openai-codex", "model": "gpt-6.1-sol"}},
        {"event_id": "other-provider", "status": "validated_candidate", "target_speaker_word_count": 3,
         "extractor": {"provider": "deepseek", "model": "deepseek-v4-pro"}},
    ]
    events = {row["event_id"]: {} for row in rows}
    with pytest.raises(ValueError, match="allow_same_provider"):
        audit._select_sample(
            rows, events=events, auditor_provider="openai-codex", auditor_model="gpt-6.1-sol",
            sample_size=1, include_event_ids={"different-model"},
        )
    selected, excluded_same_provider, excluded_same_model = audit._select_sample(
        rows, events=events, auditor_provider="openai-codex", auditor_model="gpt-6.1-sol",
        sample_size=2, include_event_ids=set(), allow_same_provider=True,
    )
    assert {row["event_id"] for row in selected} == {"different-model", "other-provider"}
    assert excluded_same_provider == []
    assert excluded_same_model == ["same-model"]


def test_audit_response_requires_exact_ids_and_source_citations():
    response = {
        "mention_coverage": 3,
        "idea_coverage": 4,
        "idea_grounding": 3,
        "idea_atomicity": 2,
        "link_quality": 4,
        "idea_reviews": [{"idea_id": "idea_a", "support": 3, "atomicity": 2, "issue_flags": ["over_split"], "reason": "A detail may be separate."}],
        "issue_examples": [{
            "target_ids": ["idea_a"], "issue_flag": "over_split", "note": "The ideas duplicate one claim.",
            "evidence": [{"turn_id": "blind:turn:0000", "exact_quote": "supports this policy"}],
        }],
        "missing_content": [],
        "overall_note": "Forecastability was not assessed without pre-event context.",
    }
    result = audit._validate(
        response,
        review_ids={"idea_a"},
        mention_ids={"mention_a"},
        idea_ids={"idea_a"},
        blind_turns={"blind:turn:0000": "The speaker supports this policy."},
    )
    assert result["idea_reviews"][0]["idea_id"] == "idea_a"
    response["issue_examples"][0]["evidence"][0]["exact_quote"] = "invented quote"
    with pytest.raises(ValueError, match="exact substring"):
        audit._validate(
            response,
            review_ids={"idea_a"},
            mention_ids={"mention_a"},
            idea_ids={"idea_a"},
            blind_turns={"blind:turn:0000": "The speaker supports this policy."},
        )


def test_idea_review_sample_is_stratified_and_bounded():
    ideas = [
        {"idea_id": f"idea_{i:02d}", "prominence": "central" if i % 2 else "brief", "forecastable": i % 3 == 0}
        for i in range(12)
    ]
    selected = audit._select_idea_ids(ideas, 5)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert set(selected) <= {idea["idea_id"] for idea in ideas}
    assert audit._select_idea_ids(ideas[:3], 5) == ["idea_00", "idea_01", "idea_02"]
