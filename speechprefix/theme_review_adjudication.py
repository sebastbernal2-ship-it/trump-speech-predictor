"""Export explicitly human-adjudicated broad theme labels from blind reviews."""

from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_theme_ranker import read_jsonl
from .theme_review_evaluation import LABELS, STANCE_LABELS, _validate_packet, evaluate_theme_review_agreement


def generate_adjudication_template(
    annotator_packet_path: str | Path,
    key_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a blank, event-ID-free decision form for a human adjudicator."""
    packets = read_jsonl(annotator_packet_path)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    item_ids = [row.get("review_item_id") for row in packets]
    key_ids = [item.get("review_item_id") for item in key.get("items", []) if isinstance(item, dict)]
    if not item_ids or len(set(item_ids)) != len(item_ids) or set(item_ids) != set(key_ids):
        raise ValueError("annotator packet and hidden key item IDs do not match")
    templates = []
    for row in packets:
        taxonomy = row.get("taxonomy")
        if not isinstance(taxonomy, list) or not taxonomy:
            raise ValueError(f"taxonomy missing for {row.get('review_item_id')}")
        templates.append({
            "review_item_id": row["review_item_id"],
            "adjudicated_by": "",
            "adjudicated_at": "",
            "family_decisions": [
                {"family_id": item["family_id"], "label": None, "evidence_turn_ids": [], "adjudication_note": ""}
                for item in taxonomy
            ],
            "event_specific_ideas": [],
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adjudication template: {output}")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in templates), encoding="utf-8")
    return {"output": str(output), "item_count": len(templates)}


def _aware_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return parsed.isoformat()


def export_adjudicated_theme_labels(
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    key_path: str | Path,
    decisions_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate human decisions against two blind reviews and export reviewed labels.

    This function does not adjudicate automatically. Every final label and
    event-specific idea must be supplied in the adjudicator's decisions file.
    """
    evaluate_theme_review_agreement(annotator_a_path, annotator_b_path, key_path)
    a_rows = read_jsonl(annotator_a_path)
    b_rows = read_jsonl(annotator_b_path)
    a = _validate_packet(a_rows, "annotator A")
    b = _validate_packet(b_rows, "annotator B")
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    key_by_id = {item["review_item_id"]: item for item in key["items"]}
    decisions = read_jsonl(decisions_path)
    decision_by_id = {row.get("review_item_id"): row for row in decisions}
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != set(a):
        raise ValueError("adjudication decisions must contain exactly one row for every review item")
    output_rows = []
    for item_id in sorted(a):
        decision = decision_by_id[item_id]
        adjudicator = decision.get("adjudicated_by")
        if not isinstance(adjudicator, str) or not adjudicator.strip():
            raise ValueError(f"adjudicated_by is required for {item_id}")
        adjudicated_at = _aware_timestamp(decision.get("adjudicated_at"), f"{item_id}.adjudicated_at")
        valid_turn_ids = {turn["turn_id"] for turn in a[item_id]["row"]["target_speaker_turns"]}
        family_decisions = decision.get("family_decisions")
        if not isinstance(family_decisions, list):
            raise ValueError(f"family_decisions must be a list for {item_id}")
        family_by_id = {item.get("family_id"): item for item in family_decisions if isinstance(item, dict)}
        if len(family_by_id) != len(family_decisions) or set(family_by_id) != set(a[item_id]["families"]):
            raise ValueError(f"adjudication must decide every fixed family exactly once for {item_id}")
        final_families = {}
        for family_id, final in family_by_id.items():
            label = final.get("label")
            citations = final.get("evidence_turn_ids")
            note = final.get("adjudication_note", "")
            if label not in LABELS or not isinstance(citations, list):
                raise ValueError(f"invalid final family decision for {item_id}/{family_id}")
            if any(citation not in valid_turn_ids for citation in citations):
                raise ValueError(f"unknown evidence turn for {item_id}/{family_id}")
            if label == "present" and not citations:
                raise ValueError(f"present final label requires evidence turns for {item_id}/{family_id}")
            if label == "absent" and citations:
                raise ValueError(f"absent final label cannot cite supporting turns for {item_id}/{family_id}")
            annotator_a = a[item_id]["families"][family_id]["label"]
            annotator_b = b[item_id]["families"][family_id]["label"]
            if (annotator_a != annotator_b or label not in {annotator_a, annotator_b}) and (
                not isinstance(note, str) or not note.strip()
            ):
                raise ValueError(f"adjudication_note is required to resolve disagreement for {item_id}/{family_id}")
            final_families[family_id] = {
                "label": label,
                "evidence_turn_ids": citations,
                "adjudication_note": note.strip() if isinstance(note, str) else "",
            }
        ideas = decision.get("event_specific_ideas")
        if not isinstance(ideas, list) or len(ideas) > 10:
            raise ValueError(f"event_specific_ideas must be a list of at most ten for {item_id}")
        final_ideas = []
        for idea in ideas:
            if not isinstance(idea, dict) or not isinstance(idea.get("idea"), str) or not idea["idea"].strip():
                raise ValueError(f"empty event-specific idea for {item_id}")
            citations = idea.get("evidence_turn_ids")
            if not isinstance(citations, list) or not citations or any(citation not in valid_turn_ids for citation in citations):
                raise ValueError(f"event-specific idea needs valid transcript citations for {item_id}")
            if idea.get("stance") not in STANCE_LABELS:
                raise ValueError(f"invalid event-specific idea stance for {item_id}")
            final_ideas.append({
                "idea": idea["idea"].strip(),
                "stance": idea["stance"],
                "evidence_turn_ids": citations,
            })
        identity = key_by_id[item_id]
        family_definitions = {
            entry["family_id"]: entry.get("definition", "")
            for entry in a[item_id]["row"]["taxonomy"]
        }
        output_rows.append({
            "schema_version": 1,
            "event_id": identity["event_id"],
            "event_date": identity["event_date"],
            "review_item_id": item_id,
            "source_record_sha256": identity["source_record_sha256"],
            "target_transcript_sha256": identity["target_transcript_sha256"],
            "annotation_policy": "independent-human-review-with-explicit-adjudication-v1",
            "review_status": "human-adjudicated",
            "family_definitions": family_definitions,
            "adjudicated_by": adjudicator.strip(),
            "adjudicated_at": adjudicated_at,
            "annotator_a_packet_sha256": hashlib.sha256(Path(annotator_a_path).read_bytes()).hexdigest(),
            "annotator_b_packet_sha256": hashlib.sha256(Path(annotator_b_path).read_bytes()).hexdigest(),
            "family_labels": final_families,
            "positive_families": sorted(family for family, value in final_families.items() if value["label"] == "present"),
            "negative_families": sorted(family for family, value in final_families.items() if value["label"] == "absent"),
            "uncertain_families": sorted(family for family, value in final_families.items() if value["label"] == "uncertain"),
            "event_specific_ideas": final_ideas,
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adjudicated label artifact: {output}")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    return {
        "output": str(output),
        "adjudicated_event_count": len(output_rows),
        "input_decisions_sha256": hashlib.sha256(Path(decisions_path).read_bytes()).hexdigest(),
        "label_policy": "human adjudication required; no model or assistant labels were copied",
    }
