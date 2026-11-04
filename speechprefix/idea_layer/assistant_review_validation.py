"""Validate provenance and ordering for transcript-only assistant outcome reviews.

This checks declared workflow metadata and content hashes; hashes cannot prove that
an annotator actually refrained from opening hidden material.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_STANCES = {"support", "oppose", "describe", "unclear"}
_ALLOWED_PROMINENCE = {"central", "repeated", "secondary", "brief"}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(body.encode("utf-8"))


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return parsed


def _read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    raw = source.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{source}: expected at least one JSONL record")
    return rows, _sha256_bytes(raw)


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value, _sha256_bytes(raw)


def _index_unique(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} record is missing nonempty {key}")
        if value in indexed:
            raise ValueError(f"duplicate {key} in {label}: {value}")
        indexed[value] = row
    return indexed


def _validate_ideas(ideas: Any, *, blind_id: str, turn_ids: set[str], mapped: bool) -> list[dict[str, Any]]:
    if not isinstance(ideas, list):
        raise ValueError(f"{blind_id}: outcome_ideas must be a list")
    ids: set[str] = set()
    for index, idea in enumerate(ideas):
        field = f"{blind_id}.outcome_ideas[{index}]"
        if not isinstance(idea, dict):
            raise ValueError(f"{field} must be an object")
        for name in ("idea_id", "idea", "stance", "prominence", "confidence", "rationale"):
            value = idea.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field}.{name} must be nonempty text")
        if idea["idea_id"] in ids:
            raise ValueError(f"{field}: duplicate idea_id {idea['idea_id']!r}")
        ids.add(idea["idea_id"])
        if idea["stance"] not in _ALLOWED_STANCES:
            raise ValueError(f"{field}.stance is unsupported: {idea['stance']!r}")
        if idea["prominence"] not in _ALLOWED_PROMINENCE:
            raise ValueError(f"{field}.prominence is unsupported: {idea['prominence']!r}")
        if idea["confidence"] not in _ALLOWED_CONFIDENCE:
            raise ValueError(f"{field}.confidence is unsupported: {idea['confidence']!r}")
        if not isinstance(idea.get("forecastable"), bool):
            raise ValueError(f"{field}.forecastable must be boolean")
        citations = idea.get("evidence_turn_ids")
        if not isinstance(citations, list) or not citations:
            raise ValueError(f"{field}.evidence_turn_ids must be a nonempty list")
        if any(not isinstance(turn_id, str) or not turn_id for turn_id in citations):
            raise ValueError(f"{field}.evidence_turn_ids must contain nonempty strings")
        unknown = set(citations) - turn_ids
        if unknown:
            raise ValueError(f"{field} cites unknown {'source' if mapped else 'blind'} turns: {sorted(unknown)}")
    return ideas


def validate_assistant_review_artifacts(
    *,
    pass_a_paths: Iterable[str | Path],
    mapped_paths: Iterable[str | Path],
    key_path: str | Path,
    events_path: str | Path,
    transcript_packet_path: str | Path,
) -> dict[str, Any]:
    """Validate paired Pass-A/mapped artifacts and their source provenance.

    Pass-A and mapped files may each contain one or many JSONL records. Every row
    must be paired by review_item_id, and its citations must map exactly through
    the sealed review key and source transcript packet.
    """
    pass_a_by_id: dict[str, tuple[dict[str, Any], str, Path]] = {}
    mapped_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
    pass_a_file_hashes: dict[Path, str] = {}
    for path_value in pass_a_paths:
        path = Path(path_value)
        rows, file_hash = _read_jsonl(path)
        pass_a_file_hashes[path.resolve()] = file_hash
        for row in rows:
            blind_id = row.get("blind_item_id")
            if not isinstance(blind_id, str) or not blind_id:
                raise ValueError(f"{path}: Pass-A row is missing blind_item_id")
            if blind_id in pass_a_by_id:
                raise ValueError(f"duplicate Pass-A review item: {blind_id}")
            pass_a_by_id[blind_id] = (row, file_hash, path)
    for path_value in mapped_paths:
        path = Path(path_value)
        rows, _ = _read_jsonl(path)
        for row in rows:
            blind_id = row.get("review_item_id")
            if not isinstance(blind_id, str) or not blind_id:
                raise ValueError(f"{path}: mapped row is missing review_item_id")
            if blind_id in mapped_by_id:
                raise ValueError(f"duplicate mapped review item: {blind_id}")
            mapped_by_id[blind_id] = (row, path)

    if not pass_a_by_id:
        raise ValueError("no Pass-A review records were provided")
    if set(pass_a_by_id) != set(mapped_by_id):
        only_a = sorted(set(pass_a_by_id) - set(mapped_by_id))
        only_mapped = sorted(set(mapped_by_id) - set(pass_a_by_id))
        raise ValueError(f"Pass-A/mapped item sets differ; only_pass_a={only_a}, only_mapped={only_mapped}")

    key, key_file_hash = _read_json(key_path)
    key_rows = key.get("items")
    if not isinstance(key_rows, list):
        raise ValueError("review key must contain an items list")
    key_by_id = _index_unique(key_rows, "review_item_id", "review key")
    event_rows, _ = _read_jsonl(events_path)
    events_by_id = _index_unique(event_rows, "event_id", "event corpus")
    transcript_rows, _ = _read_jsonl(transcript_packet_path)
    transcripts_by_id = _index_unique(transcript_rows, "review_item_id", "transcript packet")

    results: list[dict[str, Any]] = []
    for blind_id in sorted(pass_a_by_id):
        pass_a, pass_a_hash, pass_a_path = pass_a_by_id[blind_id]
        mapped, _mapped_path = mapped_by_id[blind_id]
        if pass_a.get("schema_version") != 1 or pass_a.get("artifact_type") != "assistant-reviewed-outcome-pass-a":
            raise ValueError(f"{blind_id}: invalid Pass-A schema/artifact_type")
        if not str(pass_a.get("review_status", "")).startswith("assistant-reviewed-transcript-only"):
            raise ValueError(f"{blind_id}: invalid Pass-A review_status")
        if pass_a.get("annotation_policy") != "assistant-transcript-only-review-v1":
            raise ValueError(f"{blind_id}: unexpected annotation_policy")
        review_meta = pass_a.get("review_metadata")
        if not isinstance(review_meta, dict) or review_meta.get("reviewer_type") != "AI assistant":
            raise ValueError(f"{blind_id}: missing AI reviewer provenance in Pass A")
        if review_meta.get("full_transcript_read_before_unblinding") is not True and review_meta.get(
            "full_available_transcript_read_before_unblinding"
        ) is not True:
            raise ValueError(f"{blind_id}: Pass A does not declare full available transcript review")
        if review_meta.get("event_metadata_hidden") is not True and review_meta.get(
            "pass_a_blind_to_event_identity_forecasts_news_and_weak_labels"
        ) is not True:
            raise ValueError(f"{blind_id}: Pass A does not declare event metadata hidden")
        if review_meta.get("forecasts_news_and_weak_labels_hidden") is not True and review_meta.get(
            "pass_a_blind_to_event_identity_forecasts_news_and_weak_labels"
        ) is not True:
            raise ValueError(f"{blind_id}: Pass A does not declare forecasts/news/weak labels hidden")
        protocol_hash = review_meta.get("protocol_sha256")
        if not isinstance(protocol_hash, str) or not _SHA256.fullmatch(protocol_hash):
            raise ValueError(f"{blind_id}: invalid Pass-A protocol_sha256")

        key_row = key_by_id.get(blind_id)
        transcript_row = transcripts_by_id.get(blind_id)
        if key_row is None or transcript_row is None:
            raise ValueError(f"{blind_id}: missing sealed key or blind transcript row")
        event_id = key_row.get("event_id")
        event = events_by_id.get(event_id)
        if event is None:
            raise ValueError(f"{blind_id}: event {event_id!r} missing from event corpus")
        if mapped.get("schema_version") != 1 or mapped.get("artifact_type") != "assistant-reviewed-transcript-idea-outcomes":
            raise ValueError(f"{blind_id}: invalid mapped schema/artifact_type")
        if not str(mapped.get("review_status", "")).startswith("assistant-reviewed-transcript-only"):
            raise ValueError(f"{blind_id}: invalid mapped review_status")
        if mapped.get("annotation_policy") != "assistant-transcript-only-review-v1":
            raise ValueError(f"{blind_id}: unexpected mapped annotation_policy")
        if mapped.get("event_id") != event_id or mapped.get("event_date") != key_row.get("event_date"):
            raise ValueError(f"{blind_id}: mapped event identity/date disagrees with sealed key")
        if mapped.get("source_url") != key_row.get("source_url"):
            raise ValueError(f"{blind_id}: mapped source_url disagrees with sealed key")

        actual_record_hash = _canonical_hash(event)
        if key_row.get("source_record_sha256") != actual_record_hash:
            raise ValueError(f"{blind_id}: source-record hash does not match event corpus")
        if mapped.get("source_record_sha256") != actual_record_hash:
            raise ValueError(f"{blind_id}: mapped source-record hash mismatch")
        turns = transcript_row.get("target_speaker_turns")
        if not isinstance(turns, list) or len(turns) != key_row.get("target_turn_count"):
            raise ValueError(f"{blind_id}: source turn count disagrees with sealed key")
        if any(not isinstance(turn, dict) or not isinstance(turn.get("turn_id"), str) or not isinstance(turn.get("text"), str) for turn in turns):
            raise ValueError(f"{blind_id}: malformed target-speaker turns")
        actual_transcript_hash = _sha256_bytes(
            "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns).encode("utf-8")
        )
        if key_row.get("target_transcript_sha256") != actual_transcript_hash:
            raise ValueError(f"{blind_id}: source transcript hash does not match sealed key")
        if mapped.get("target_transcript_sha256") != actual_transcript_hash:
            raise ValueError(f"{blind_id}: mapped transcript hash mismatch")
        blind_turn_ids = {f"{blind_id}:turn:{index:04d}" for index in range(len(turns))}
        blind_digest = _sha256_bytes(
            "\n".join(
                f"{blind_id}:turn:{index:04d}\t{turn['text']}" for index, turn in enumerate(turns)
            ).encode("utf-8")
        )
        if pass_a.get("target_transcript_blind_sha256") != blind_digest:
            raise ValueError(f"{blind_id}: Pass-A blind transcript hash mismatch")

        ideas_a = _validate_ideas(pass_a.get("outcome_ideas"), blind_id=blind_id, turn_ids=blind_turn_ids, mapped=False)
        source_turn_ids = {turn["turn_id"] for turn in turns}
        ideas_mapped = _validate_ideas(mapped.get("outcome_ideas"), blind_id=blind_id, turn_ids=source_turn_ids, mapped=True)
        expected_mapped = deepcopy(ideas_a)
        blind_to_source = {
            f"{blind_id}:turn:{index:04d}": turn["turn_id"] for index, turn in enumerate(turns)
        }
        for idea in expected_mapped:
            idea["evidence_turn_ids"] = [blind_to_source[citation] for citation in idea["evidence_turn_ids"]]
        if expected_mapped != ideas_mapped:
            raise ValueError(f"{blind_id}: mapped outcomes differ from frozen Pass A beyond citation mapping")

        actual_pass_hash = pass_a_file_hashes[pass_a_path.resolve()]
        if mapped.get("pass_a_blind_artifact_sha256") != actual_pass_hash:
            raise ValueError(f"{blind_id}: mapped Pass-A artifact hash mismatch")
        if mapped.get("review_key_sha256") != key_file_hash:
            raise ValueError(f"{blind_id}: mapped review-key hash mismatch")
        mapped_reviewer = mapped.get("reviewer")
        method = mapped.get("review_method")
        if not isinstance(mapped_reviewer, dict) or mapped_reviewer.get("reviewer_type") != "AI assistant":
            raise ValueError(f"{blind_id}: missing mapped AI reviewer provenance")
        if mapped_reviewer.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"{blind_id}: Pass-A/mapped protocol hashes differ")
        if not isinstance(method, dict):
            raise ValueError(f"{blind_id}: review_method must be an object")
        if method.get("candidate_forecasts_opened") is not False:
            raise ValueError(f"{blind_id}: candidate forecasts were not declared unopened")
        if method.get("forecasts_news_and_weak_labels_opened", False) is not False:
            raise ValueError(f"{blind_id}: forecasts/news/weak labels were not declared unopened")
        if method.get("outcome_labels_frozen_before_unblinding") is not True:
            raise ValueError(f"{blind_id}: outcome labels were not declared frozen before unblinding")
        unblind_at = _timestamp(mapped.get("unblinded_for_provenance_at"), f"{blind_id}.unblinded_for_provenance_at")
        reviewed_at = _timestamp(review_meta.get("reviewed_at"), f"{blind_id}.reviewed_at")
        if reviewed_at >= unblind_at:
            raise ValueError(f"{blind_id}: Pass A timestamp is not earlier than unblinding timestamp")
        if mapped_reviewer.get("reviewed_at"):
            mapped_reviewed_at = _timestamp(mapped_reviewer["reviewed_at"], f"{blind_id}.mapped_reviewed_at")
            if mapped_reviewed_at < reviewed_at or mapped_reviewed_at > unblind_at:
                raise ValueError(f"{blind_id}: reviewer timestamp falls outside Pass-A-to-unblinding window")

        forecastable_count = sum(idea["forecastable"] for idea in ideas_mapped)
        results.append({
            "review_item_id": blind_id,
            "event_id": event_id,
            "event_date": key_row.get("event_date"),
            "outcome_count": len(ideas_mapped),
            "forecastable_count": forecastable_count,
            "nonforecastable_count": len(ideas_mapped) - forecastable_count,
            "cited_turn_count": sum(len(idea["evidence_turn_ids"]) for idea in ideas_mapped),
            "duplicate_citation_count": sum(
                len(idea["evidence_turn_ids"]) - len(set(idea["evidence_turn_ids"]))
                for idea in ideas_mapped
            ),
            "source_hashes_valid": True,
            "citation_mapping_valid": True,
            "pass_a_precedes_unblinding_by_seconds": (unblind_at - reviewed_at).total_seconds(),
            "pass_a_file_sha256": actual_pass_hash,
            "mapped_file": str(_mapped_path),
        })

    return {
        "schema_version": 1,
        "artifact_type": "assistant-review-validation-report",
        "validation_scope": "declared workflow metadata, immutable file hashes, source provenance, and exact turn-citation mapping",
        "cryptographic_limitation": "Hashes validate artifact integrity but cannot prove that hidden forecasts or labels were not viewed.",
        "item_count": len(results),
        "outcome_count": sum(item["outcome_count"] for item in results),
        "forecastable_outcome_count": sum(item["forecastable_count"] for item in results),
        "items": results,
    }
