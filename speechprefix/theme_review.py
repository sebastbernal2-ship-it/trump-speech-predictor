"""Generate blinded broad-theme annotation packets from local transcript records."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .context_theme_ranker import read_jsonl
from .headline_idea_matching import ISSUE_PHRASE_EXTRACTOR_VERSION, issue_phrase_lexicon_sha256


INSTRUCTIONS = (
    "Annotate only the target speaker's ideas. First, without consulting the taxonomy below, list up to five "
    "distinct event-specific ideas in neutral wording with supporting turn IDs and stance "
    "(support, oppose, describe, unclear). Do not infer an idea from external news or another speaker. "
    "After recording those ideas, mark every fixed theme family present, absent, or uncertain using the speaker's "
    "meaning rather than a keyword alone; cite supporting turns for present families. Do not inspect model predictions "
    "or context headlines while annotating."
)


def _write_new(path: str | Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing annotation artifact: {output}")
    output.write_text(content, encoding="utf-8")


def generate_theme_review_packets(
    snapshots_path: str | Path,
    event_records_path: str | Path,
    labels_path: str | Path,
    *,
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    key_path: str | Path,
    seed: int = 17,
    selection_description: str = "User-provided event selection; see the local selection manifest.",
) -> dict[str, Any]:
    """Create two transcript-only forms plus a local event-ID key.

    The family-label file is read only for the fixed taxonomy. No draft
    positives, candidate rankings, context headlines, dates, titles, or URLs are
    copied into the annotator packets.
    """
    if not isinstance(selection_description, str) or not selection_description.strip():
        raise ValueError("selection_description must be nonempty")
    snapshots = read_jsonl(snapshots_path)
    source_records = read_jsonl(event_records_path)
    label_records = read_jsonl(labels_path)
    event_ids = [row.get("event_id") for row in snapshots]
    if not event_ids or len(set(event_ids)) != len(event_ids):
        raise ValueError("selection manifest must contain unique event IDs")
    selection_by_id = {row["event_id"]: row for row in snapshots}
    source_by_id = {row.get("event_id"): row for row in source_records}
    if len(source_by_id) != len(source_records):
        raise ValueError("duplicate source event IDs")
    labels_by_id = {row.get("event_id"): row for row in label_records}
    definitions = label_records[0].get("family_definitions") if label_records else None
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("a fixed family taxonomy is required")
    if any(row.get("event_id") not in labels_by_id for row in snapshots):
        raise ValueError("every snapshot event must have a taxonomy record")
    signature = json.dumps(definitions, ensure_ascii=False, sort_keys=True)
    if any(json.dumps(row.get("family_definitions"), ensure_ascii=False, sort_keys=True) != signature for row in label_records):
        raise ValueError("theme taxonomy differs across label records")

    ordered_ids = sorted(event_ids, key=lambda event_id: (
        selection_by_id[event_id].get("event_date", ""), event_id,
    ))
    random.Random(seed).shuffle(ordered_ids)
    packet_rows, key_rows = [], []
    for index, event_id in enumerate(ordered_ids, 1):
        record = source_by_id.get(event_id)
        if not isinstance(record, dict):
            raise ValueError(f"missing transcript source record for {event_id}")
        selection_date = selection_by_id[event_id].get("event_date")
        if selection_date != record.get("event_date"):
            raise ValueError(f"selection manifest event date differs from transcript source for {event_id}")
        target = str(record.get("target_speaker", "")).casefold()
        turns = []
        for turn in record.get("turns", []):
            is_target = turn.get("speaker_role") == "target_speaker" or (
                target and str(turn.get("speaker_name", "")).casefold() == target
            )
            text = turn.get("text") or turn.get("raw_text")
            if is_target and isinstance(text, str) and text.strip():
                turns.append({"turn_id": turn.get("turn_id"), "text": text.strip()})
        if not turns:
            raise ValueError(f"no target-speaker transcript turns for {event_id}")
        item_id = f"blind-item-{index:03d}"
        transcript_digest = hashlib.sha256(
            "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns).encode("utf-8")
        ).hexdigest()
        packet_rows.append({
            "schema_version": 1,
            "review_item_id": item_id,
            "instructions": INSTRUCTIONS,
            "event_specific_ideas": [],
            "taxonomy": [
                {"family_id": family, "definition": definitions[family], "label": None, "evidence_turn_ids": []}
                for family in sorted(definitions)
            ],
            "target_speaker_turns": turns,
            "annotation_status": "blank_for_independent_human_review",
        })
        key_rows.append({
            "review_item_id": item_id,
            "event_id": event_id,
            "event_date": selection_date,
            "source_url": record.get("source_url"),
            "source_record_sha256": hashlib.sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "target_transcript_sha256": transcript_digest,
            "target_turn_count": len(turns),
        })
    packet_a = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet_rows)
    packet_b = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet_rows)
    key = {
        "schema_version": 2,
        "artifact_type": "blind-theme-review-key",
        "seed": seed,
        "selection_description": selection_description,
        "selection_manifest_sha256": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
        "issue_phrase_extractor": {
            "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
            "lexicon_sha256": issue_phrase_lexicon_sha256(),
        },
        "event_records_sha256": hashlib.sha256(Path(event_records_path).read_bytes()).hexdigest(),
        "taxonomy_source_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        "items": key_rows,
        "limitations": [
            "This small sample is an annotation and coverage pilot, not a semantic gold set or a population prevalence estimate.",
            "Reviewers must not see the hidden key, assistant drafts, model predictions, or context snapshots.",
            "Theme-family absences are explicit annotation fields; event-specific ideas are free-response and require adjudication.",
        ],
    }
    key_text = json.dumps(key, ensure_ascii=False, indent=2) + "\n"
    outputs = [Path(annotator_a_path), Path(annotator_b_path), Path(key_path)]
    if len(set(outputs)) != len(outputs):
        raise ValueError("annotation packet and key paths must be distinct")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing annotation artifact: {path}")
    _write_new(annotator_a_path, packet_a)
    _write_new(annotator_b_path, packet_b)
    _write_new(key_path, key_text)
    return {"item_count": len(packet_rows), "family_count": len(definitions), "seed": seed}
