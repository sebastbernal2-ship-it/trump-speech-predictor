import pytest

from speechprefix.idea_layer.transcript_annotations import TranscriptAnnotationError, validate_transcript_annotation


def sample():
    return {
        "artifact_type": "transcript-annotation", "schema_version": 1, "event_id": "event-1",
        "coverage": "full_pass", "annotation_pass": {"method": "assistant_first_pass", "review_status": "pending_user_review"},
        "mentions": [{"mention_id": "m1", "evidence": [{"turn_id": "t1", "exact_quote": "We support this."}]}],
        "ideas": [{"idea_id": "i1", "occurrence": "the stated policy issue"}],
        "key_points": [{"key_point_id": "k1", "detail": "a specific policy detail"}],
        "topics": [{"topic_id": "topic1", "label": "public policy"}],
        "opinions": [{"opinion_id": "o1", "text": "The speaker supports the proposal.", "attribution": "speaker", "stance": "supports", "uncertainty": "low"}],
        "emotions": [{"emotion_id": "e1", "label": "supportive tone", "uncertainty": "medium"}],
        "idea_mentions": [{"mention_id": "m1", "idea_id": "i1"}],
        "key_point_ideas": [{"key_point_id": "k1", "idea_id": "i1"}],
        "topic_ideas": [{"topic_id": "topic1", "idea_id": "i1"}],
        "opinion_targets": [{"opinion_id": "o1", "target_type": "key_point", "target_id": "k1"}],
        "opinion_mentions": [{"opinion_id": "o1", "mention_id": "m1"}],
        "key_point_mentions": [{"key_point_id": "k1", "mention_id": "m1"}],
        "emotion_mentions": [{"emotion_id": "e1", "mention_id": "m1"}],
    }


def test_validates_linked_layers_and_exact_quote():
    result = validate_transcript_annotation(sample(), {"t1": "We support this. It is important."})
    assert result["validation"]["structural_valid"]
    assert result["validation"]["semantic_entailment_validated"] is False


def test_rejects_quote_not_in_cited_target_turn():
    with pytest.raises(TranscriptAnnotationError, match="exact transcript substring"):
        validate_transcript_annotation(sample(), {"t1": "Different words."})


def test_rejects_broken_relation():
    record = sample()
    record["opinion_targets"][0]["target_id"] = "missing"
    with pytest.raises(TranscriptAnnotationError, match="unresolved target_id"):
        validate_transcript_annotation(record, {"t1": "We support this."})
