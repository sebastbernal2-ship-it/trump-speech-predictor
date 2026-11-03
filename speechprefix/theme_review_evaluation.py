"""Validate independent theme annotations and report inter-annotator agreement."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_theme_ranker import read_jsonl

LABELS = {"present", "absent", "uncertain"}
STANCE_LABELS = {"support", "oppose", "describe", "unclear"}
BLINDING_FIELDS = {"event_id", "event_date", "title", "source_url", "candidate_sources", "assistant_draft_targets"}


def _matrix_and_stats(left: list[str], right: list[str], labels: list[str]) -> dict[str, Any]:
    n = len(left)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(left, right):
        matrix[a][b] += 1
    observed = sum(matrix[label][label] for label in labels) / n if n else 0.0
    expected = sum(
        (sum(matrix[label].values()) / n) * (sum(matrix[row][label] for row in labels) / n)
        for label in labels
    ) if n else 0.0
    kappa = (observed - expected) / (1.0 - expected) if n and expected < 1.0 else None
    positive_left = sum(a == "present" for a in left)
    positive_right = sum(b == "present" for b in right)
    both_positive = sum(a == b == "present" for a, b in zip(left, right))
    negative_left = sum(a == "absent" for a in left)
    negative_right = sum(b == "absent" for b in right)
    both_negative = sum(a == b == "absent" for a, b in zip(left, right))
    return {
        "n_items": n,
        "observed_agreement": observed,
        "cohen_kappa": kappa,
        "positive_agreement": 2 * both_positive / (positive_left + positive_right) if positive_left + positive_right else None,
        "negative_agreement": 2 * both_negative / (negative_left + negative_right) if negative_left + negative_right else None,
        "annotator_a_distribution": {label: sum(value == label for value in left) for label in labels},
        "annotator_b_distribution": {label: sum(value == label for value in right) for label in labels},
        "confusion_matrix_a_rows_b_columns": matrix,
    }


def _transcript_digest(turns: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_packet(rows: list[dict[str, Any]], annotator: str) -> dict[str, dict[str, Any]]:
    mapped = {}
    for row in rows:
        item_id = row.get("review_item_id")
        if not isinstance(item_id, str) or not item_id or item_id in mapped:
            raise ValueError(f"{annotator} packet has a missing or duplicate review_item_id")
        if BLINDING_FIELDS.intersection(row):
            raise ValueError(f"{annotator} packet reveals event identity or model/context labels")
        if row.get("annotation_status") != "completed":
            raise ValueError(f"{annotator} annotation is incomplete for {item_id}")
        turns = row.get("target_speaker_turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{annotator} packet has no transcript turns for {item_id}")
        turn_ids = [turn.get("turn_id") for turn in turns if isinstance(turn, dict)]
        if len(turn_ids) != len(turns) or any(not isinstance(turn_id, str) or not turn_id for turn_id in turn_ids) or len(set(turn_ids)) != len(turn_ids):
            raise ValueError(f"{annotator} has invalid or duplicate transcript turn IDs for {item_id}")
        valid_turn_ids = set(turn_ids)
        taxonomy = row.get("taxonomy")
        if not isinstance(taxonomy, list) or not taxonomy:
            raise ValueError(f"{annotator} taxonomy is missing for {item_id}")
        families = {}
        for entry in taxonomy:
            if not isinstance(entry, dict) or not isinstance(entry.get("family_id"), str):
                raise ValueError(f"{annotator} has an invalid taxonomy entry for {item_id}")
            family = entry["family_id"]
            label = entry.get("label")
            citations = entry.get("evidence_turn_ids")
            if family in families or label not in LABELS or not isinstance(citations, list):
                raise ValueError(f"{annotator} has an invalid label for {item_id}/{family}")
            if any(citation not in valid_turn_ids for citation in citations):
                raise ValueError(f"{annotator} cites an unknown transcript turn for {item_id}/{family}")
            if label == "present" and not citations:
                raise ValueError(f"present label requires supporting turn citations: {item_id}/{family}")
            if label == "absent" and citations:
                raise ValueError(f"absent label cannot have supporting turn citations: {item_id}/{family}")
            families[family] = {"label": label, "citations": citations}
        ideas = row.get("event_specific_ideas")
        if not isinstance(ideas, list) or len(ideas) > 5:
            raise ValueError(f"{annotator} event_specific_ideas must be a list of at most five for {item_id}")
        for idea in ideas:
            if not isinstance(idea, dict) or not isinstance(idea.get("idea"), str) or not idea["idea"].strip():
                raise ValueError(f"{annotator} has an empty event-specific idea for {item_id}")
            citations = idea.get("evidence_turn_ids")
            if not isinstance(citations, list) or not citations or any(citation not in valid_turn_ids for citation in citations):
                raise ValueError(f"{annotator} event-specific idea needs valid turn citations for {item_id}")
            if idea.get("stance") not in STANCE_LABELS:
                raise ValueError(f"{annotator} event-specific idea has an invalid stance for {item_id}")
        mapped[item_id] = {"row": row, "families": families, "ideas": ideas}
    return mapped


def evaluate_theme_review_agreement(
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    key_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate completed blind packets and calculate family-level agreement."""
    a_rows, b_rows = read_jsonl(annotator_a_path), read_jsonl(annotator_b_path)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if not isinstance(key, dict) or key.get("artifact_type") != "blind-theme-review-key":
        raise ValueError("invalid hidden annotation key")
    a, b = _validate_packet(a_rows, "annotator A"), _validate_packet(b_rows, "annotator B")
    if not a or set(a) != set(b):
        raise ValueError("annotator packets must contain the same nonempty review items")
    key_items = key.get("items")
    if not isinstance(key_items, list):
        raise ValueError("hidden key has no item mappings")
    key_by_id = {item.get("review_item_id"): item for item in key_items if isinstance(item, dict)}
    key_event_ids = [item.get("event_id") for item in key_items if isinstance(item, dict)]
    if len(key_by_id) != len(key_items) or set(key_by_id) != set(a) or len(set(key_event_ids)) != len(key_event_ids):
        raise ValueError("hidden key and annotation items do not match")
    family_sets = {tuple(sorted(item["families"])) for item in a.values()}
    family_sets_b = {tuple(sorted(item["families"])) for item in b.values()}
    if len(family_sets) != 1 or family_sets != family_sets_b:
        raise ValueError("annotators must use the same fixed family taxonomy")
    family_ids = list(next(iter(family_sets)))
    family_results = {}
    disagreements = {family: [] for family in family_ids}
    event_specific_differences = []
    for family in family_ids:
        labels_a, labels_b = [], []
        for item_id in sorted(a):
            left, right = a[item_id], b[item_id]
            for packet in (left["row"], right["row"]):
                if packet.get("instructions") != a[item_id]["row"].get("instructions"):
                    raise ValueError(f"annotation instructions differ for {item_id}")
                if packet.get("target_speaker_turns") != a[item_id]["row"].get("target_speaker_turns"):
                    raise ValueError(f"transcript content differs for {item_id}")
                if [(item.get("family_id"), item.get("definition")) for item in packet["taxonomy"]] != [
                    (item.get("family_id"), item.get("definition")) for item in a[item_id]["row"]["taxonomy"]
                ]:
                    raise ValueError(f"theme taxonomy differs between annotators for {item_id}")
            if _transcript_digest(left["row"]["target_speaker_turns"]) != key_by_id[item_id].get("target_transcript_sha256"):
                raise ValueError(f"transcript fingerprint mismatch for {item_id}")
            labels_a.append(left["families"][family]["label"])
            labels_b.append(right["families"][family]["label"])
            if labels_a[-1] != labels_b[-1]:
                disagreements[family].append(key_by_id[item_id]["event_id"])
        family_results[family] = _matrix_and_stats(labels_a, labels_b, ["present", "absent", "uncertain"])
    for item_id in sorted(a):
        ideas_a = a[item_id]["ideas"]
        ideas_b = b[item_id]["ideas"]
        if ideas_a != ideas_b:
            event_specific_differences.append(key_by_id[item_id]["event_id"])
    all_labels_a = [a[item_id]["families"][family]["label"] for item_id in sorted(a) for family in family_ids]
    all_labels_b = [b[item_id]["families"][family]["label"] for item_id in sorted(b) for family in family_ids]
    report = {
        "schema_version": 1,
        "report_type": "independent-theme-annotation-agreement",
        "label_policy": "independent human annotations; free-text ideas remain unadjudicated",
        "input_sha256": {
            "annotator_a": hashlib.sha256(Path(annotator_a_path).read_bytes()).hexdigest(),
            "annotator_b": hashlib.sha256(Path(annotator_b_path).read_bytes()).hexdigest(),
            "hidden_key": hashlib.sha256(Path(key_path).read_bytes()).hexdigest(),
        },
        "item_count": len(a),
        "family_count": len(family_ids),
        "overall_family_agreement": _matrix_and_stats(all_labels_a, all_labels_b, ["present", "absent", "uncertain"]),
        "per_family": family_results,
        "family_disagreement_event_ids": disagreements,
        "event_specific_idea_disagreement_event_ids_exact_text_only": event_specific_differences,
        "adjudication_required": sorted(set(event_id for event_ids in disagreements.values() for event_id in event_ids) | set(event_specific_differences)),
        "limitations": [
            "Cohen's kappa can be unstable on a small pilot and is sensitive to label prevalence.",
            "Exact free-text comparison is not semantic agreement; all event-specific ideas need human alignment/adjudication.",
            "Agreement does not establish that the taxonomy is valid or that labels are representative.",
            "The supplied hidden key and transcript hashes are provenance checks, not proof annotators were blinded in practice.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing agreement report: {output}")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
