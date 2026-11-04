"""Deterministic source-family ablations for frozen idea-layer snapshots."""

from __future__ import annotations
import hashlib
import json
from typing import Any

from .contracts import validate_idea_input_snapshot

ABLATIONS = frozenset({"combined", "news_only", "event_context_only", "speaker_history_only"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def derive_source_ablation(snapshot: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return a content-addressed snapshot retaining event metadata plus one source family."""
    validate_idea_input_snapshot(snapshot)
    if variant not in ABLATIONS:
        raise ValueError(f"variant must be one of {sorted(ABLATIONS)}")
    source_by_id = {source["source_id"]: source for source in snapshot["sources"]}
    manifest = json.loads(json.dumps(snapshot["event_manifest"], ensure_ascii=False))
    metadata_ids = set(manifest["event_metadata_source_ids"])

    def include(source: dict[str, Any]) -> bool:
        if source["source_id"] in metadata_ids:
            return True
        source_type = source["source_type"]
        is_history = source.get("source_role") == "speaker_history"
        if variant == "combined":
            return source_type in {"news", "other_primary_source"}
        if variant == "news_only":
            return source_type == "news"
        if variant == "speaker_history_only":
            return is_history
        return False

    selected = {source_id: source for source_id, source in source_by_id.items() if include(source)}
    retained_ids = set(selected)
    facts = manifest.get("known_context", [])
    manifest["known_context"] = [
        fact for fact in facts if set(fact.get("source_ids", [])) <= retained_ids
    ]
    manifest_required = metadata_ids
    missing = manifest_required - retained_ids
    if missing:
        raise ValueError(f"snapshot omits mandatory event metadata sources: {sorted(missing)}")

    derived = {
        key: value for key, value in snapshot.items()
        if key not in {"snapshot_sha256", "sources", "input_source_ids", "event_manifest", "event_manifest_sha256", "source_ablation"}
    }
    manifest_bytes = _canonical(manifest).encode("utf-8")
    derived.update({
        "event_manifest": manifest,
        "event_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_source_ids": sorted(retained_ids),
        "sources": sorted(
            selected.values(),
            key=lambda row: (row["observed_at"], row["source_id"]),
        ),
        "source_ablation": variant,
    })
    derived["snapshot_sha256"] = hashlib.sha256(_canonical(derived).encode("utf-8")).hexdigest()
    validate_idea_input_snapshot(derived)
    return derived
