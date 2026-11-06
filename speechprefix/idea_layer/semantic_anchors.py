"""Separate direct-expression anchor artifacts over frozen outcome layers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .transcript_outcomes import validate_layered_outcome

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ANCHOR_ID = re.compile(r"^anchor_[0-9a-f]{16}$")
_ARTIFACT_TYPE = "layered-idea-expression-anchors"
_RECORD_FIELDS = {
    "schema_version", "artifact_type", "event_id", "source_outcome_sha256",
    "source_record_sha256", "target_transcript_sha256", "anchor_set_sha256",
    "anchors", "unanchored_idea_ids", "extractor", "validation",
}
_VALIDATION = {
    "structural_valid": True,
    "source_outcome_bound": True,
    "every_idea_accounted_for": True,
    "semantic_entailment_validated": False,
    "anchor_completeness_validated": False,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def anchor_id(mention_id: str, idea_id: str) -> str:
    return "anchor_" + canonical_sha256([mention_id, idea_id])[:16]


def anchor_set_projection(anchors: list[dict[str, Any]], unanchored_idea_ids: list[str]) -> dict[str, Any]:
    return {
        "anchors": [
            {"mention_id": item["mention_id"], "idea_id": item["idea_id"]}
            for item in anchors
        ],
        "unanchored_idea_ids": list(unanchored_idea_ids),
    }


def build_expression_anchor_record(
    outcome: dict[str, Any],
    proposed_anchors: list[dict[str, Any]],
    unanchored_idea_ids: list[str],
    *,
    provider: str,
    model: str,
    engine_version: str,
    prompt_version: str,
    turns_by_id: dict[str, str],
    response_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a provenance-bound anchor set; abstentions are explicit per idea."""
    validate_layered_outcome(outcome, turns_by_id)
    if not isinstance(proposed_anchors, list) or not isinstance(unanchored_idea_ids, list):
        raise ValueError("anchors and unanchored_idea_ids must be lists")
    anchors = []
    for item in proposed_anchors:
        if not isinstance(item, dict) or set(item) != {"mention_id", "idea_id"}:
            raise ValueError("each anchor must contain exactly mention_id and idea_id")
        mention_id, idea_id = item["mention_id"], item["idea_id"]
        if not isinstance(mention_id, str) or not isinstance(idea_id, str):
            raise ValueError("anchor endpoint IDs must be strings")
        anchors.append({"anchor_id": anchor_id(mention_id, idea_id), "mention_id": mention_id, "idea_id": idea_id})
    anchors.sort(key=lambda item: (item["idea_id"], item["mention_id"]))
    if any(not isinstance(idea_id, str) for idea_id in unanchored_idea_ids):
        raise ValueError("unanchored idea IDs must be strings")
    unanchored = sorted(unanchored_idea_ids)
    projection = anchor_set_projection(anchors, unanchored)
    record = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "event_id": outcome.get("event_id"),
        "source_outcome_sha256": canonical_sha256(outcome),
        "source_record_sha256": outcome.get("source_record_sha256"),
        "target_transcript_sha256": outcome.get("target_transcript_sha256"),
        "anchor_set_sha256": canonical_sha256(projection),
        "anchors": anchors,
        "unanchored_idea_ids": unanchored,
        "extractor": {
            "provider": provider,
            "model": model,
            "engine_version": engine_version,
            "prompt_version": prompt_version,
            **({"response_sha256": response_sha256} if response_sha256 is not None else {}),
        },
        "validation": dict(_VALIDATION),
    }
    return validate_expression_anchor_record(record, outcome, turns_by_id)


def validate_expression_anchor_record(
    record: dict[str, Any],
    outcome: dict[str, Any],
    turns_by_id: dict[str, str],
    *,
    max_anchors: int = 1200,
) -> dict[str, Any]:
    """Validate anchor endpoints and provenance without claiming semantic truth."""
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError(f"anchor record must contain exactly: {sorted(_RECORD_FIELDS)}")
    if record.get("schema_version") != 1 or record.get("artifact_type") != _ARTIFACT_TYPE:
        raise ValueError("unsupported expression-anchor schema")
    if record.get("event_id") != outcome.get("event_id"):
        raise ValueError("anchor event_id does not match its source outcome")
    if record.get("source_outcome_sha256") != canonical_sha256(outcome):
        raise ValueError("anchor source outcome hash mismatch")
    for field in ("source_outcome_sha256", "source_record_sha256", "target_transcript_sha256", "anchor_set_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"anchor {field} must be a lowercase SHA-256 digest")
    for field in ("source_record_sha256", "target_transcript_sha256"):
        if record.get(field) != outcome.get(field):
            raise ValueError(f"anchor {field} does not match its source outcome")
    validate_layered_outcome(outcome, turns_by_id)

    mention_ids = {item["mention_id"] for item in outcome["mentions"]}
    idea_ids = {item["idea_id"] for item in outcome["ideas"]}
    anchors, unanchored = record.get("anchors"), record.get("unanchored_idea_ids")
    if not isinstance(anchors, list) or len(anchors) > max_anchors:
        raise ValueError(f"anchors must be a list of at most {max_anchors}")
    if not isinstance(unanchored, list) or any(not isinstance(idea_id, str) for idea_id in unanchored):
        raise ValueError("unanchored_idea_ids must be a list of strings")
    if unanchored != sorted(set(unanchored)):
        raise ValueError("unanchored_idea_ids must be a sorted unique list")
    if any(idea_id not in idea_ids for idea_id in unanchored):
        raise ValueError("unanchored_idea_ids must resolve to source ideas")

    seen_pairs: set[tuple[str, str]] = set()
    seen_anchor_ids: set[str] = set()
    anchored_ideas: set[str] = set()
    normalized_pairs = []
    for index, item in enumerate(anchors):
        field = f"anchors[{index}]"
        if not isinstance(item, dict) or set(item) != {"anchor_id", "mention_id", "idea_id"}:
            raise ValueError(f"{field} must contain exactly anchor_id, mention_id, and idea_id")
        mention_id, idea_id, identifier = item["mention_id"], item["idea_id"], item["anchor_id"]
        if not isinstance(mention_id, str) or mention_id not in mention_ids:
            raise ValueError(f"{field}.mention_id does not resolve in the source outcome")
        if not isinstance(idea_id, str) or idea_id not in idea_ids:
            raise ValueError(f"{field}.idea_id does not resolve in the source outcome")
        pair = (mention_id, idea_id)
        if pair in seen_pairs:
            raise ValueError(f"{field} duplicates an expression anchor pair")
        seen_pairs.add(pair)
        anchored_ideas.add(idea_id)
        expected_id = anchor_id(mention_id, idea_id)
        if not isinstance(identifier, str) or not _ANCHOR_ID.fullmatch(identifier) or identifier != expected_id:
            raise ValueError(f"{field}.anchor_id is not deterministic for its endpoints")
        if identifier in seen_anchor_ids:
            raise ValueError(f"{field}.anchor_id is duplicated")
        seen_anchor_ids.add(identifier)
        normalized_pairs.append({"mention_id": mention_id, "idea_id": idea_id})
    if anchored_ideas & set(unanchored):
        raise ValueError("an idea cannot be both anchored and explicitly unanchored")
    if anchored_ideas | set(unanchored) != idea_ids:
        raise ValueError("each source idea must have one or more anchors or be explicitly unanchored")
    normalized_pairs.sort(key=lambda item: (item["idea_id"], item["mention_id"]))
    projection = {"anchors": normalized_pairs, "unanchored_idea_ids": unanchored}
    if record.get("anchor_set_sha256") != canonical_sha256(projection):
        raise ValueError("anchor-set hash mismatch")

    extractor = record.get("extractor")
    if not isinstance(extractor, dict) or set(extractor) - {"provider", "model", "engine_version", "prompt_version", "response_sha256"}:
        raise ValueError("extractor provenance has unsupported fields")
    for field in ("provider", "model", "engine_version", "prompt_version"):
        if not isinstance(extractor.get(field), str) or not extractor[field].strip():
            raise ValueError(f"extractor.{field} must be nonempty")
    if "response_sha256" in extractor and (
        not isinstance(extractor["response_sha256"], str) or not _HASH.fullmatch(extractor["response_sha256"])
    ):
        raise ValueError("extractor.response_sha256 must be a lowercase SHA-256 digest")
    if record.get("validation") != _VALIDATION:
        raise ValueError("validation flags must distinguish structural from semantic validation")
    return record
