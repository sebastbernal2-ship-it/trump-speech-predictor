"""Append-only source ledger and immutable pre-speech input snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .contracts import (
    is_source_eligible_at,
    validate_event_manifest,
    validate_idea_input_snapshot,
    validate_source_evidence,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {source}:{line_number}")
            rows.append(row)
    return rows


def read_source_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the local source ledger; duplicate IDs are corruption."""
    rows = _read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        source_id = validate_source_evidence(row)["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate source_id in ledger: {source_id}")
        seen.add(source_id)
    return rows


def append_source_items(incoming: list[dict[str, Any]], ledger_path: str | Path) -> dict[str, int]:
    """Validate and append source objects, treating identical IDs idempotently."""
    ledger_file = Path(ledger_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in incoming:
        source_id = validate_source_evidence(row)["source_id"]
        prior = by_id.get(source_id)
        if prior is not None and _canonical(prior) != _canonical(row):
            raise ValueError(f"conflicting duplicate source_id in import batch: {source_id}")
        by_id[source_id] = row
    existing_rows = read_source_ledger(ledger_file)
    existing = {row["source_id"]: row for row in existing_rows}
    additions = []
    skipped = 0
    for source_id, row in by_id.items():
        prior = existing.get(source_id)
        if prior is not None:
            if _canonical(prior) != _canonical(row):
                raise ValueError(f"source_id already exists with different content: {source_id}")
            skipped += 1
        else:
            additions.append(row)
    if additions:
        ledger_file.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in additions)
        descriptor = os.open(ledger_file, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            raise
    return {"incoming_unique": len(by_id), "appended": len(additions), "already_present_identically": skipped}


def append_source_records(input_path: str | Path, ledger_path: str | Path) -> dict[str, int]:
    """Validate and append JSONL source items; identical IDs are idempotent."""
    input_file, ledger_file = Path(input_path), Path(ledger_path)
    if input_file.resolve() == ledger_file.resolve():
        raise ValueError("source import input and ledger paths must differ")
    return append_source_items(_read_jsonl(input_file), ledger_file)


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field} must be a timezone-aware RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def freeze_idea_input_snapshot(
    manifest_path: str | Path,
    ledger_path: str | Path,
    output_path: str | Path,
    *,
    lookback_days: int = 30,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    """Freeze currently verified event/news/post inputs before the declared cutoff."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    manifest_file, output = Path(manifest_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite input snapshot: {output}")
    try:
        manifest_bytes = manifest_file.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"invalid event manifest JSON: {manifest_file}") from None
    event = validate_event_manifest(manifest)
    cutoff = _parse_time(event["prediction_cutoff_at"], "prediction_cutoff_at")
    created = frozen_at or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("frozen_at must be timezone-aware")
    created = created.astimezone(timezone.utc)
    if created >= cutoff:
        raise ValueError("cannot freeze event inputs at or after prediction cutoff")

    ledger = read_source_ledger(ledger_path)
    by_id = {row["source_id"]: row for row in ledger}
    required = set(manifest["event_metadata_source_ids"])
    for fact in manifest.get("known_context", []):
        required.update(fact["source_ids"])
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"event manifest references sources missing from ledger: {missing}")
    wrong_metadata_types = sorted(
        source_id for source_id in manifest["event_metadata_source_ids"]
        if by_id[source_id]["source_type"] != "event_metadata"
    )
    if wrong_metadata_types:
        raise ValueError(f"event_metadata_source_ids must reference event_metadata records: {wrong_metadata_types}")

    lookback_start = cutoff - timedelta(days=lookback_days)
    selected: dict[str, dict[str, Any]] = {}
    for source_id, source in by_id.items():
        try:
            eligible = is_source_eligible_at(source, cutoff.isoformat())
        except ValueError:
            if source_id in required:
                raise
            continue
        observed = _parse_time(source["observed_at"], "observed_at")
        published = _parse_time(source["published_at"], "published_at")
        if observed > created:
            eligible = False
        if source_id in required and not eligible:
            raise ValueError(f"required event context source is not verified before cutoff: {source_id}")
        if eligible and (
            published >= lookback_start
            or source_id in required
            or source.get("source_role") == "speaker_history"
        ):
            selected[source_id] = source

    required_sources = {source_id: selected[source_id] for source_id in required}
    required_canonical_ids = {
        row.get("canonical_source_id", source_id) for source_id, row in required_sources.items()
    }
    optional_by_canonical: dict[str, tuple[datetime, str, dict[str, Any]]] = {}
    for source_id, source in selected.items():
        if source_id in required:
            continue
        canonical_id = source.get("canonical_source_id", source_id)
        if canonical_id in required_canonical_ids:
            continue
        prior = optional_by_canonical.get(canonical_id)
        observed = _parse_time(source["observed_at"], "observed_at")
        if prior is None or observed > prior[0]:
            optional_by_canonical[canonical_id] = (observed, source_id, source)
    selected = dict(required_sources)
    selected.update({source_id: source for _, source_id, source in optional_by_canonical.values()})

    frozen_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "idea-layer-input-snapshot",
        "event_id": event["event_id"],
        "prediction_cutoff_at": event["prediction_cutoff_at"],
        "snapshot_created_at": created.isoformat(),
        "lookback_days": lookback_days,
        "event_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "event_manifest": frozen_manifest,
        "input_source_ids": sorted(selected),
        "sources": sorted(
            selected.values(), key=lambda row: (_parse_time(row["observed_at"], "observed_at"), row["source_id"]),
        ),
    }
    canonical = _canonical(snapshot)
    snapshot["snapshot_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    validation = validate_idea_input_snapshot(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite input snapshot: {output}") from None
    return {
        "output": str(output),
        "event_id": validation["event_id"],
        "source_count": validation["source_count"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "strict_pre_cutoff": True,
    }
