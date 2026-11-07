from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "classify_speech_key_points",
    ROOT / "research/idea_layers/points/classify_speech_key_points.py",
)
assert _SPEC is not None and _SPEC.loader is not None
classifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(classifier)


def _row():
    return {
        "mentions": [
            {"mention_id": "m1", "text": "He rejects the impeachment process.", "evidence": [{"exact_quote": "I reject impeachment."}]},
            {"mention_id": "m2", "text": "He disputes the Ukraine call account.", "evidence": [{"exact_quote": "The call was perfect."}]},
        ],
        "ideas": [
            {"idea_id": "impeachment", "concept": "Impeachment response", "summary": "He opposes impeachment as an attempt to reverse his election.", "kind": "political", "stance": "oppose", "prominence": "central", "forecastable": True},
            {"idea_id": "ukraine_call", "concept": "Ukraine call dispute", "summary": "He denies wrongdoing in his conversation with Ukraine's president.", "kind": "political", "stance": "oppose", "prominence": "repeated", "forecastable": True},
        ],
        "links": [
            {"mention_id": "m1", "idea_id": "impeachment", "relation": "expresses"},
            {"mention_id": "m2", "idea_id": "ukraine_call", "relation": "expresses"},
        ],
    }


def test_prompt_preserves_ids_and_distinct_story_points():
    prompt = classifier.build_prompt(_row())
    assert "do not rewrite, merge, split, add, or delete ideas" in prompt
    assert "Keep all distinct key points as separate nodes" in prompt
    assert "Preserve the distinction between the legislative impeachment process" in prompt


def test_validation_accepts_separate_roots_and_parent_detail():
    row = _row()
    result = classifier.validate({"annotations": [
        {"idea_id": "impeachment", "status": "key_point", "parent_idea_id": None, "parent_relation": None, "rationale": "Independently developed political argument."},
        {"idea_id": "ukraine_call", "status": "key_point", "parent_idea_id": None, "parent_relation": None, "rationale": "A distinct event-specific dispute."},
    ]}, row)
    assert len(result["annotations"]) == 2


def test_validation_rejects_missing_duplicate_or_misparented_ideas():
    row = _row()
    one = {"idea_id": "impeachment", "status": "key_point", "parent_idea_id": None, "parent_relation": None, "rationale": "Distinct point."}
    with pytest.raises(ValueError, match="omitted IDs"):
        classifier.validate({"annotations": [one]}, row)
    with pytest.raises(ValueError, match="duplicate idea_id"):
        classifier.validate({"annotations": [one, one]}, row)
    invalid = [
        one,
        {"idea_id": "ukraine_call", "status": "supporting_detail", "parent_idea_id": "impeachment", "parent_relation": "evidence_for", "rationale": "Fixture only."},
    ]
    # This structure is valid: the validator checks form, not semantic entailment.
    assert len(classifier.validate({"annotations": invalid}, row)["annotations"]) == 2
    invalid_parent = [
        one,
        {"idea_id": "ukraine_call", "status": "key_point", "parent_idea_id": "impeachment", "parent_relation": "evidence_for", "rationale": "Wrong relation type."},
    ]
    with pytest.raises(ValueError, match="specific_to"):
        classifier.validate({"annotations": invalid_parent}, row)
    cycle = [
        {"idea_id": "impeachment", "status": "key_point", "parent_idea_id": "ukraine_call", "parent_relation": "specific_to", "rationale": "Cycle fixture."},
        {"idea_id": "ukraine_call", "status": "key_point", "parent_idea_id": "impeachment", "parent_relation": "specific_to", "rationale": "Cycle fixture."},
    ]
    with pytest.raises(ValueError, match="cycle"):
        classifier.validate({"annotations": cycle}, row)
