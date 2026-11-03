"""Prospective context-snapshot validation and legacy-pilot audit."""

from __future__ import annotations
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field} must be an RFC 3339 timestamp with a time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _validate_available_source(source: Any, field: str, cutoff: datetime, frozen: datetime) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"{field} must be an object")
    if not isinstance(source.get("source_url"), str) or not source["source_url"].strip():
        raise ValueError(f"{field}.source_url is required")
    published = _timestamp(source.get("published_at"), f"{field}.published_at")
    observed = _timestamp(source.get("observed_at"), f"{field}.observed_at")
    if published >= cutoff:
        raise ValueError(f"{field} was not published before the prediction cutoff")
    if observed > frozen:
        raise ValueError(f"{field} was observed after the snapshot was frozen")
    if field.startswith("event_metadata") and source.get("availability_verified_before_cutoff") is not True:
        raise ValueError(f"{field} availability must be independently verified")


def validate_prospective_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate a source-frozen v2 snapshot suitable for prospective evaluation.

    This checks recorded provenance and chronology; it cannot independently
    establish that the supplied timestamps or availability attestations are true.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be an object")
    if snapshot.get("schema_version") != 2 or snapshot.get("protocol") != "prospective-freeze-v2":
        raise ValueError("prospective snapshots must use schema_version 2 and prospective-freeze-v2")
    forbidden = {"transcript", "transcript_text", "target_text", "target_ideas", "theme_labels", "draft_themes", "outcome_labels"}
    if forbidden.intersection(snapshot):
        raise ValueError("prospective context snapshot must not include transcript or target-label fields")
    event_id = snapshot.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id is required")
    cutoff = _timestamp(snapshot.get("prediction_cutoff_at"), "prediction_cutoff_at")
    event_start = _timestamp(snapshot.get("event_start_at"), "event_start_at")
    frozen = _timestamp(snapshot.get("snapshot_frozen_at"), "snapshot_frozen_at")
    if frozen > cutoff:
        raise ValueError("snapshot_frozen_at must not be after prediction_cutoff_at")
    if cutoff >= event_start:
        raise ValueError("prediction_cutoff_at must be before event_start_at")

    metadata = snapshot.get("event_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("event_metadata provenance is required")
    _validate_available_source(metadata, "event_metadata", cutoff, frozen)
    location = snapshot.get("location_metadata")
    if location is not None:
        if not isinstance(location, dict) or location.get("availability_verified_before_cutoff") is not True:
            raise ValueError("location metadata may be used only with verified pre-cutoff availability")
        _validate_available_source(location, "location_metadata", cutoff, frozen)

    queries = snapshot.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("at least one timestamped retrieval query is required")
    for index, query in enumerate(queries):
        if not isinstance(query, dict) or not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError(f"queries[{index}].query is required")
        retrieved = _timestamp(query.get("retrieved_at"), f"queries[{index}].retrieved_at")
        if retrieved > frozen:
            raise ValueError(f"queries[{index}] was retrieved after the snapshot was frozen")
        if retrieved > cutoff:
            raise ValueError(f"queries[{index}] was retrieved after the prediction cutoff")

    sources = snapshot.get("candidate_sources")
    if not isinstance(sources, list):
        raise ValueError("candidate_sources must be a list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("title"), str) or not source["title"].strip():
            raise ValueError(f"candidate_sources[{index}].title is required")
        if not isinstance(source.get("publisher"), str) or not source["publisher"].strip():
            raise ValueError(f"candidate_sources[{index}].publisher is required")
        _validate_available_source(source, f"candidate_sources[{index}]", cutoff, frozen)
    digest = snapshot.get("snapshot_sha256")
    if digest is not None:
        payload = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest != expected:
            raise ValueError("snapshot_sha256 does not match snapshot content")
    return {"event_id": event_id, "candidate_source_count": len(sources), "strict_prospective_eligible": True}


def seal_prospective_snapshot(snapshot: dict[str, Any], *, frozen_at: datetime | None = None) -> dict[str, Any]:
    """Set freeze time from the current clock, validate, and attach a content hash."""
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be an object")
    sealed = {key: value for key, value in snapshot.items() if key not in {"snapshot_sha256", "snapshot_frozen_at"}}
    frozen = frozen_at or datetime.now(timezone.utc)
    if frozen.tzinfo is None or frozen.utcoffset() is None:
        raise ValueError("frozen_at must include a timezone")
    sealed.update({
        "schema_version": 2,
        "protocol": "prospective-freeze-v2",
        "snapshot_frozen_at": frozen.isoformat(),
    })
    validate_prospective_snapshot(sealed)
    cutoff = _timestamp(sealed["prediction_cutoff_at"], "prediction_cutoff_at")
    if frozen >= cutoff:
        raise ValueError("cannot seal a snapshot at or after its prediction cutoff")
    canonical = json.dumps(sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sealed["snapshot_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return sealed


def audit_legacy_snapshots(snapshots_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Audit the exploratory v1 snapshots for strict prospective-freeze requirements."""
    source_path = Path(snapshots_path)
    rows = []
    with source_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSONL at {source_path}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {source_path}:{line_number}")
            rows.append(row)
    event_results = []
    for row in rows:
        event_id = row.get("event_id")
        event_date = date.fromisoformat(row["event_date"])
        issues = []
        if row.get("schema_version") != 2 or row.get("context_retrieval_protocol") != "prospective-freeze-v2":
            issues.append("legacy_exploratory_protocol")
        if not row.get("event_start_at"):
            issues.append("exact_event_start_time_missing")
        if not row.get("prediction_cutoff_at"):
            issues.append("exact_prediction_cutoff_missing")
        retrieved_at = row.get("retrieved_at")
        try:
            retrieved_date = _timestamp(retrieved_at, "retrieved_at").date() if retrieved_at else None
        except ValueError:
            retrieved_date = None
        if retrieved_date is None:
            issues.append("snapshot_retrieval_timestamp_missing_or_invalid")
        elif retrieved_date >= event_date:
            issues.append("snapshot_retrieved_on_or_after_event_date")
        if row.get("location_hint") and row.get("location_availability_verified_before_cutoff") is not True:
            issues.append("location_availability_unverified")
        sources = row.get("candidate_sources", [])
        missing_observation_time = sum(not isinstance(item, dict) or not item.get("observed_at") for item in sources)
        if missing_observation_time:
            issues.append("candidate_source_observation_times_missing")
        date_only_sources = sum(
            isinstance(item, dict) and item.get("date_precision") != "timestamp"
            for item in sources
        )
        if date_only_sources:
            issues.append("candidate_publication_times_only_day_precision")
        event_results.append({
            "event_id": event_id,
            "event_date": event_date.isoformat(),
            "candidate_source_count": len(sources),
            "issues": sorted(set(issues)),
            "strict_prospective_eligible": not issues,
        })
    issue_counts: dict[str, int] = {}
    for result in event_results:
        for issue in result["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    report = {
        "schema_version": 1,
        "audit_type": "legacy-context-snapshot-prospective-eligibility",
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "snapshot_count": len(rows),
        "strict_prospective_eligible_count": sum(result["strict_prospective_eligible"] for result in event_results),
        "issue_event_counts": issue_counts,
        "events": event_results,
        "limitations": [
            "This audit checks recorded fields, not the truth of source timestamps or retrospective-selection claims.",
            "A date-only headline date does not independently prove that the source was publicly available at the prediction cutoff.",
            "Legacy snapshots are exploratory and must not be relabeled as prospective forecasts.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing audit artifact: {output}")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
