"""Strict, JSON-compatible contracts for the standalone idea forecasting layer."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any


SOURCE_TYPES = frozenset({"news", "trump_post", "event_metadata", "other_primary_source"})
AVAILABILITY_STATES = frozenset({"verified_before_cutoff", "not_verified", "unknown"})
UPDATE_ACTIONS = frozenset({"added", "raised", "lowered", "removed", "unchanged"})
_FORBIDDEN_FORECAST_KEYS = frozenset({
    "transcript", "transcript_text", "target_text", "target_ideas", "theme_labels",
    "draft_themes", "outcome_labels", "post_event_notes", "future_turns", "target_summary",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _required_text(record: dict[str, Any], key: str, field: str | None = None) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field or key} must be a nonempty string")
    return value.strip()


def seal_pre_speech_forecast(forecast: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a SHA-256 digest over its complete content."""
    sealed = json.loads(json.dumps(forecast, ensure_ascii=False))
    sealed.pop("forecast_sha256", None)
    canonical = json.dumps(sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sealed["forecast_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return sealed


def _reject_forbidden_keys(value: Any, field: str = "record") -> None:
    if isinstance(value, dict):
        normalized_keys = {key.casefold().replace("-", "_").replace(" ", "_") for key in value}
        forbidden = _FORBIDDEN_FORECAST_KEYS.intersection(normalized_keys)
        if forbidden:
            raise ValueError(f"{field} contains forbidden outcome fields: {sorted(forbidden)}")
        for key, nested in value.items():
            _reject_forbidden_keys(nested, f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{field}[{index}]")


def validate_source_evidence(source: dict[str, Any]) -> dict[str, str]:
    """Validate one source item while preserving its identity and time provenance."""
    if not isinstance(source, dict):
        raise ValueError("source evidence must be an object")
    if source.get("schema_version") != 1 or source.get("record_type") != "idea-layer-source":
        raise ValueError("source evidence must use schema_version 1 and record_type idea-layer-source")
    source_id = _required_text(source, "source_id")
    if "canonical_source_id" in source:
        _required_text(source, "canonical_source_id")
    if "archive_sha256" in source and (
        not isinstance(source["archive_sha256"], str) or not _SHA256.fullmatch(source["archive_sha256"])
    ):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 hex digest")
    source_type = source.get("source_type")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {source_type!r}")
    _required_text(source, "source_url")
    _required_text(source, "publisher")
    _required_text(source, "content")
    _required_text(source, "reuse_terms")
    digest = source.get("content_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")
    expected_digest = hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
    if digest != expected_digest:
        raise ValueError("content_sha256 does not match source content")
    published = _timestamp(source.get("published_at"), "published_at")
    observed = _timestamp(source.get("observed_at"), "observed_at")
    if published > observed:
        raise ValueError("observed_at cannot precede published_at")
    availability_status = source.get("availability_status")
    if availability_status not in AVAILABILITY_STATES:
        raise ValueError("availability_status must explicitly record verification status")
    if availability_status == "verified_before_cutoff":
        _required_text(source, "availability_evidence")
        _required_text(source, "timestamp_basis")
    if source_type == "trump_post":
        _required_text(source, "account")
        _required_text(source, "post_id")
    return {"source_id": source_id, "published_at": published.isoformat(), "observed_at": observed.isoformat()}


def is_source_eligible_at(source: dict[str, Any], cutoff_at: str) -> bool:
    """Return true only for sourced information captured and verified before cutoff."""
    validate_source_evidence(source)
    cutoff = _timestamp(cutoff_at, "cutoff_at")
    published = _timestamp(source["published_at"], "published_at")
    observed = _timestamp(source["observed_at"], "observed_at")
    return (
        published < cutoff
        and observed < cutoff
        and source.get("availability_status") == "verified_before_cutoff"
    )


def validate_event_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    """Validate event metadata known before speech start and its forecast cutoff."""
    if not isinstance(manifest, dict):
        raise ValueError("event manifest must be an object")
    _reject_forbidden_keys(manifest, "event_manifest")
    if manifest.get("schema_version") != 1 or manifest.get("record_type") != "idea-layer-event":
        raise ValueError("event manifest must use schema_version 1 and record_type idea-layer-event")
    event_id = _required_text(manifest, "event_id")
    event_start = _timestamp(manifest.get("event_start_at"), "event_start_at")
    cutoff = _timestamp(manifest.get("prediction_cutoff_at"), "prediction_cutoff_at")
    if cutoff >= event_start:
        raise ValueError("prediction_cutoff_at must be strictly before event_start_at")
    metadata_source_ids = manifest.get("event_metadata_source_ids")
    if (
        not isinstance(metadata_source_ids, list)
        or not metadata_source_ids
        or any(not isinstance(item, str) or not item for item in metadata_source_ids)
        or len(set(metadata_source_ids)) != len(metadata_source_ids)
    ):
        raise ValueError("event_metadata_source_ids must be a nonempty list of unique source IDs")
    facts = manifest.get("known_context", [])
    if not isinstance(facts, list):
        raise ValueError("known_context must be a list")
    fact_ids = set()
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise ValueError(f"known_context[{index}] must be an object")
        fact_id = _required_text(fact, "fact_id", f"known_context[{index}].fact_id")
        _required_text(fact, "fact", f"known_context[{index}].fact")
        if fact_id in fact_ids:
            raise ValueError(f"duplicate known_context fact_id: {fact_id}")
        fact_ids.add(fact_id)
        source_ids = fact.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids or any(not isinstance(item, str) or not item for item in source_ids):
            raise ValueError(f"known_context[{index}].source_ids must be a nonempty list of IDs")
    return {"event_id": event_id, "event_start_at": event_start.isoformat(), "prediction_cutoff_at": cutoff.isoformat()}


def validate_idea_input_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate a frozen, content-addressed bundle of pre-cutoff event inputs."""
    if not isinstance(snapshot, dict):
        raise ValueError("input snapshot must be an object")
    _reject_forbidden_keys(snapshot, "input_snapshot")
    if snapshot.get("schema_version") != 1 or snapshot.get("record_type") != "idea-layer-input-snapshot":
        raise ValueError("input snapshot must use schema_version 1 and record_type idea-layer-input-snapshot")
    manifest = snapshot.get("event_manifest")
    event = validate_event_manifest(manifest)
    if snapshot.get("event_id") != event["event_id"]:
        raise ValueError("snapshot event_id does not match event manifest")
    cutoff = _timestamp(event["prediction_cutoff_at"], "prediction_cutoff_at")
    created = _timestamp(snapshot.get("snapshot_created_at"), "snapshot_created_at")
    if created >= cutoff:
        raise ValueError("input snapshot must be frozen strictly before prediction cutoff")
    sources = snapshot.get("sources")
    input_ids = snapshot.get("input_source_ids")
    if not isinstance(sources, list) or not isinstance(input_ids, list):
        raise ValueError("snapshot sources and input_source_ids must be lists")
    source_by_id = {}
    canonical_ids = set()
    for source in sources:
        validated = validate_source_evidence(source)
        source_id = validated["source_id"]
        if source_id in source_by_id:
            raise ValueError(f"duplicate source_id in snapshot: {source_id}")
        canonical_id = source.get("canonical_source_id", source_id)
        if canonical_id in canonical_ids:
            raise ValueError(f"duplicate canonical source in snapshot: {canonical_id}")
        canonical_ids.add(canonical_id)
        source_by_id[source_id] = source
        observed = _timestamp(source["observed_at"], "observed_at")
        if observed > created:
            raise ValueError(f"source {source_id} was observed after the snapshot was frozen")
        if not is_source_eligible_at(source, cutoff.isoformat()):
            raise ValueError(f"source {source_id} is not verified and available before cutoff")
    if (
        any(not isinstance(item, str) or not item for item in input_ids)
        or len(set(input_ids)) != len(input_ids)
        or set(input_ids) != set(source_by_id)
    ):
        raise ValueError("input_source_ids must exactly match the unique source records in the snapshot")
    required_ids = set(manifest["event_metadata_source_ids"])
    for fact in manifest.get("known_context", []):
        required_ids.update(fact["source_ids"])
    if not required_ids <= set(input_ids):
        raise ValueError("snapshot omits source evidence for event metadata or known context")
    wrong_metadata_types = sorted(
        source_id for source_id in manifest["event_metadata_source_ids"]
        if source_by_id[source_id]["source_type"] != "event_metadata"
    )
    if wrong_metadata_types:
        raise ValueError(f"event_metadata_source_ids must reference event_metadata records: {wrong_metadata_types}")
    digest = snapshot.get("snapshot_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("snapshot_sha256 must be a lowercase SHA-256 hex digest")
    unhashed = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    canonical = json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError("snapshot_sha256 does not match snapshot content")
    return {"event_id": event["event_id"], "source_count": len(sources), "strict_pre_cutoff": True}


def validate_pre_speech_forecast(
    forecast: dict[str, Any], manifest: dict[str, Any], sources: list[dict[str, Any]],
    *, input_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a frozen forecast against only pre-cutoff, traceable source items."""
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be an object")
    _reject_forbidden_keys(forecast, "forecast")
    if forecast.get("schema_version") != 1 or forecast.get("record_type") != "idea-layer-forecast":
        raise ValueError("forecast must use schema_version 1 and record_type idea-layer-forecast")
    _required_text(forecast, "forecast_id")
    snapshot_digest = forecast.get("input_snapshot_sha256")
    if not isinstance(snapshot_digest, str) or not _SHA256.fullmatch(snapshot_digest):
        raise ValueError("input_snapshot_sha256 must be a lowercase SHA-256 hex digest")
    _required_text(forecast, "proposal_set_id")
    _required_text(forecast, "proposal_generator")
    proposal_digest = forecast.get("proposal_set_sha256")
    if not isinstance(proposal_digest, str) or not _SHA256.fullmatch(proposal_digest):
        raise ValueError("proposal_set_sha256 must be a lowercase SHA-256 hex digest")
    event = validate_event_manifest(manifest)
    if forecast.get("event_id") != event["event_id"]:
        raise ValueError("forecast event_id does not match event manifest")
    cutoff = _timestamp(forecast.get("prediction_cutoff_at"), "prediction_cutoff_at")
    manifest_cutoff = _timestamp(event["prediction_cutoff_at"], "manifest prediction_cutoff_at")
    if cutoff != manifest_cutoff:
        raise ValueError("forecast cutoff must exactly match event manifest")
    snapshot_created = _timestamp(forecast.get("input_snapshot_created_at"), "input_snapshot_created_at")
    if snapshot_created >= cutoff:
        raise ValueError("input snapshot must be created strictly before prediction cutoff")
    generated = _timestamp(forecast.get("generated_at"), "generated_at")
    if generated < snapshot_created:
        raise ValueError("forecast cannot predate its input snapshot")
    if generated >= cutoff:
        raise ValueError("forecast must be generated strictly before its prediction cutoff")
    if input_snapshot is not None:
        snapshot_info = validate_idea_input_snapshot(input_snapshot)
        if forecast.get("input_snapshot_sha256") != input_snapshot.get("snapshot_sha256"):
            raise ValueError("forecast input_snapshot_sha256 does not match supplied snapshot")
        if forecast.get("input_snapshot_created_at") != input_snapshot.get("snapshot_created_at"):
            raise ValueError("forecast input snapshot creation time does not match supplied snapshot")
        if snapshot_info["event_id"] != event["event_id"]:
            raise ValueError("supplied input snapshot belongs to a different event")
    _required_text(forecast, "model_version")
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        validated = validate_source_evidence(source)
        source_id = validated["source_id"]
        if source_id in source_by_id:
            raise ValueError(f"duplicate source_id: {source_id}")
        source_by_id[source_id] = source
    if input_snapshot is not None:
        snapshot_sources = {row["source_id"]: row for row in input_snapshot["sources"]}
        if source_by_id != snapshot_sources:
            raise ValueError("supplied source evidence does not match the input snapshot")
    input_ids = forecast.get("input_source_ids")
    if (
        not isinstance(input_ids, list)
        or any(not isinstance(item, str) or not item for item in input_ids)
        or len(set(input_ids)) != len(input_ids)
    ):
        raise ValueError("input_source_ids must be a list of unique source IDs")
    if any(source_id not in source_by_id for source_id in input_ids):
        raise ValueError("forecast references a source absent from the supplied source ledger")
    ineligible = [source_id for source_id in input_ids if not is_source_eligible_at(source_by_id[source_id], cutoff.isoformat())]
    if ineligible:
        raise ValueError(f"forecast includes unavailable or unverified pre-cutoff sources: {ineligible}")
    required_manifest_sources = set(manifest["event_metadata_source_ids"])
    for fact in manifest.get("known_context", []):
        required_manifest_sources.update(fact["source_ids"])
    if not required_manifest_sources <= set(input_ids):
        raise ValueError("forecast must include every source backing the event metadata and known context")
    wrong_metadata_types = sorted(
        source_id for source_id in manifest["event_metadata_source_ids"]
        if source_by_id[source_id]["source_type"] != "event_metadata"
    )
    if wrong_metadata_types:
        raise ValueError(f"event_metadata_source_ids must reference event_metadata records: {wrong_metadata_types}")
    candidates = forecast.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    seen_ids, seen_ranks = set(), set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        idea_id = _required_text(candidate, "idea_id", f"candidates[{index}].idea_id")
        _required_text(candidate, "idea", f"candidates[{index}].idea")
        if idea_id in seen_ids:
            raise ValueError(f"duplicate candidate idea_id: {idea_id}")
        seen_ids.add(idea_id)
        rank = candidate.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1 or rank in seen_ranks:
            raise ValueError(f"candidates[{index}].rank must be a unique positive integer")
        seen_ranks.add(rank)
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"candidates[{index}].score must be finite numeric")
        score_kind = candidate.get("score_kind")
        if score_kind not in {"ranking_score", "probability"}:
            raise ValueError(f"candidates[{index}].score_kind must be ranking_score or probability")
        if score_kind == "probability":
            if not 0 <= score <= 1 or not isinstance(candidate.get("calibration_ref"), str):
                raise ValueError("probability scores require [0,1] values and a calibration_ref")
        evidence_ids = candidate.get("evidence_source_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(not isinstance(item, str) or not item for item in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            raise ValueError(f"candidates[{index}].evidence_source_ids must be unique nonempty source IDs")
        if not set(evidence_ids) <= set(input_ids):
            raise ValueError(f"candidates[{index}] cites sources outside forecast inputs")
    digest = forecast.get("forecast_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("forecast_sha256 must be a lowercase SHA-256 hex digest")
    unhashed = {key: value for key, value in forecast.items() if key != "forecast_sha256"}
    canonical = json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
        raise ValueError("forecast_sha256 does not match forecast content")
    return {
        "event_id": event["event_id"],
        "candidate_count": len(candidates),
        "input_source_count": len(input_ids),
        "strict_pre_cutoff": True,
    }


def validate_idea_update(update: dict[str, Any], visible_turn_ids: list[str]) -> dict[str, Any]:
    """Validate one append-only update using only turn IDs visible at that moment."""
    if not isinstance(update, dict):
        raise ValueError("idea update must be an object")
    if update.get("schema_version") != 1 or update.get("record_type") != "idea-layer-update":
        raise ValueError("idea update must use schema_version 1 and record_type idea-layer-update")
    _required_text(update, "event_id")
    _required_text(update, "forecast_id")
    update_at = _timestamp(update.get("updated_at"), "updated_at")
    observed = update.get("observed_turn_ids")
    if (
        not isinstance(observed, list)
        or not observed
        or any(not isinstance(item, str) or not item for item in observed)
        or len(set(observed)) != len(observed)
    ):
        raise ValueError("observed_turn_ids must be a nonempty list of unique IDs")
    visible = set(visible_turn_ids)
    if not set(observed) <= visible:
        raise ValueError("update cites turns that were not yet visible")
    changes = update.get("changes")
    if not isinstance(changes, list):
        raise ValueError("changes must be a list")
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            raise ValueError(f"changes[{index}] must be an object")
        _required_text(change, "idea_id", f"changes[{index}].idea_id")
        if change.get("action") not in UPDATE_ACTIONS:
            raise ValueError(f"changes[{index}].action is invalid")
        refs = change.get("evidence_turn_ids", [])
        if (
            not isinstance(refs, list)
            or any(not isinstance(item, str) or not item for item in refs)
            or not set(refs) <= set(observed)
        ):
            raise ValueError(f"changes[{index}] cites an unobserved turn")
    return {"event_id": update["event_id"], "updated_at": update_at.isoformat(), "change_count": len(changes)}
