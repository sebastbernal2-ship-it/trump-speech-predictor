from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "group_speech_ideas_into_topics",
    ROOT / "research/idea_layers/topics/group_speech_ideas_into_topics.py",
)
assert _SPEC is not None and _SPEC.loader is not None
topics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(topics)


def _row():
    return {
        "ideas": [
            {"idea_id": "impeachment", "concept": "Impeachment", "summary": "The speaker opposes impeachment.", "kind": "political", "prominence": "central"},
            {"idea_id": "ukraine_call", "concept": "Ukraine call", "summary": "The speaker denies wrongdoing in the Ukraine call.", "kind": "political", "prominence": "repeated"},
            {"idea_id": "immigration", "concept": "Immigration policy", "summary": "The speaker supports stricter immigration enforcement.", "kind": "policy", "prominence": "central"},
        ]
    }


def test_topic_prompt_preserves_points_and_separate_claims():
    prompt = topics.build_prompt(_row())
    assert "Do not merge, split, rename, omit, add, or rewrite point nodes" in prompt
    assert "topic membership is organizational only" in prompt.lower()
    assert "A topic may be a broad umbrella for distinct claims" in prompt
    assert "Every point ID must belong to exactly one topic group" in prompt


def test_validator_accepts_partition_without_mutating_point_ids():
    row = _row()
    result = topics.validate({"topics": [
        {"topic_id": "political_process", "label": "Political process", "summary": "Claims about impeachment and Ukraine.", "member_idea_ids": ["impeachment", "ukraine_call"]},
        {"topic_id": "immigration_policy", "label": "Immigration policy", "summary": "Arguments about immigration enforcement.", "member_idea_ids": ["immigration"]},
    ]}, row)
    assert len(result["topics"]) == 2
    assert [item["idea_id"] for item in result["topic_memberships"]] == ["immigration", "impeachment", "ukraine_call"]
    assert set(item["idea_id"] for item in result["topic_memberships"]) == {idea["idea_id"] for idea in row["ideas"]}


def test_validator_rejects_omissions_duplicates_unknown_ids_and_duplicate_topics():
    row = _row()
    one = {"topic_id": "topic_a", "label": "A", "summary": "A topic.", "member_idea_ids": ["impeachment"]}
    with pytest.raises(topics.TopicLayerError, match="omitted"):
        topics.validate({"topics": [one]}, row)
    with pytest.raises(topics.TopicLayerError, match="multiple primary topics"):
        topics.validate({"topics": [
            one,
            {"topic_id": "topic_b", "label": "B", "summary": "Another.", "member_idea_ids": ["impeachment", "ukraine_call", "immigration"]},
        ]}, row)
    with pytest.raises(topics.TopicLayerError, match="unknown"):
        topics.validate({"topics": [
            {"topic_id": "topic_a", "label": "A", "summary": "A topic.", "member_idea_ids": ["impeachment", "ukraine_call", "unknown"]},
            {"topic_id": "topic_b", "label": "B", "summary": "Another.", "member_idea_ids": ["immigration"]},
        ]}, row)
    with pytest.raises(topics.TopicLayerError, match="topic_id"):
        topics.validate({"topics": [
            one,
            {"topic_id": "topic_a", "label": "B", "summary": "Another.", "member_idea_ids": ["ukraine_call", "immigration"]},
        ]}, row)
