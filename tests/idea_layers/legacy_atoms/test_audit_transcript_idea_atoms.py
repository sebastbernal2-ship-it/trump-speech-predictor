from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/idea_layers/legacy_atoms/audit_transcript_idea_atoms.py"
SPEC = importlib.util.spec_from_file_location("audit_transcript_idea_atoms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _response() -> dict:
    return {
        "event_completeness": 3,
        "major_omissions": [],
        "atom_reviews": [{
            "idea_id": "trade_policy",
            "evidence_support": 4,
            "atomicity": 3,
            "summary_stance_fidelity": 4,
            "type_forecastability_fidelity": 3,
            "issue_flags": [],
            "reason": "The cited turns directly support this trade-policy concept.",
        }],
        "overall_note": "The inventory is mostly supported with one modest coverage gap.",
    }


def test_validate_requires_exact_atom_ids_and_0_to_4_ratings():
    result = audit._validate(_response(), {"trade_policy"}, {"blind:turn:0000"})
    assert result["atom_reviews"][0]["evidence_support"] == 4
    invalid = _response()
    invalid["atom_reviews"][0]["idea_id"] = "missing_atom"
    with pytest.raises(ValueError, match="exactly match"):
        audit._validate(invalid, {"trade_policy"}, {"blind:turn:0000"})
    invalid = _response()
    invalid["atom_reviews"][0]["atomicity"] = 5
    with pytest.raises(ValueError, match="0 to 4"):
        audit._validate(invalid, {"trade_policy"}, {"blind:turn:0000"})


def test_prepare_atoms_blinds_turn_ids_and_hides_extractor_confidence():
    atoms, atom_ids, evidence_count = audit._prepare_atoms(
        [{
            "idea_id": "concept_a",
            "concept": "Concept A",
            "confidence": "high",
            "prominence": "high",
            "evidence": [{"turn_id": "source-turn-7", "exact_quote": "a supported phrase"}],
        }],
        source_to_blind={"source-turn-7": "case:turn:0000"},
        original_turns={"source-turn-7": "Here is a supported phrase in context."},
    )
    assert atom_ids == {"concept_a"}
    assert evidence_count == 1
    assert "confidence" not in atoms[0]
    assert "prominence" not in atoms[0]
    assert atoms[0]["evidence"][0]["turn_id"] == "case:turn:0000"


def test_validate_rejects_unknown_issue_flags():
    invalid = _response()
    invalid["atom_reviews"][0]["issue_flags"] = ["looks_bad"]
    with pytest.raises(ValueError, match="unsupported value"):
        audit._validate(invalid, {"trade_policy"}, {"blind:turn:0000"})


def test_event_score_uses_documented_weights():
    review = {
        "event_completeness": 4,
        "atom_reviews": [{
            "evidence_support": 4,
            "atomicity": 4,
            "summary_stance_fidelity": 4,
            "type_forecastability_fidelity": 4,
        }],
    }
    assert audit._event_score(review, citation_integrity=100.0) == pytest.approx(100.0)
    assert audit._event_score(review, citation_integrity=0.0) == pytest.approx(95.0)
