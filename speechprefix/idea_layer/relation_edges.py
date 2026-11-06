"""Separate, provenance-bound candidate relations layered over frozen ideas.

Direct-expression anchors live in a separate source-bound artifact. This module
validates optional finer-grained mention-to-idea edges without changing them.
Validation is structural only; it cannot establish semantic relation quality.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .semantic_anchors import canonical_sha256 as anchor_canonical_sha256, validate_expression_anchor_record
from .transcript_outcomes import LINK_RELATIONS, validate_layered_outcome

FINE_RELATIONS = frozenset(LINK_RELATIONS - {"expresses"})
FINE_RELATION_DEFINITIONS = {
    "supports": "The mention gives a reason, evidence, or substantive basis for the idea; mere topical relevance is insufficient.",
    "illustrates": "The mention is a concrete instance or example of the idea, not merely another detail about it.",
    "elaborates": "The mention adds a material specification or development of the idea without functioning mainly as evidence, an example, or a caveat.",
    "qualifies": "The mention narrows, limits, conditions, corrects, or materially nuances the idea.",
    "contextualizes": "The mention gives relevant setting or background but does not directly express, support, instantiate, expand, or qualify the idea.",
}
_EDGE_ID = re.compile(r"^edge_[0-9a-f]{16}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE = "layered-idea-relation-candidates"
_RECORD_FIELDS = {
    "schema_version", "artifact_type", "event_id", "source_outcome_sha256",
    "source_record_sha256", "target_transcript_sha256", "source_anchor_artifact_sha256",
    "anchor_set_sha256", "direct_expression_anchors", "unanchored_idea_ids",
    "extractor", "edges", "validation",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def direct_expression_anchors(anchor_record: dict[str, Any]) -> list[dict[str, str]]:
    """Return the frozen direct-expression anchor projection from its own artifact."""
    anchors = [
        {"mention_id": item["mention_id"], "idea_id": item["idea_id"]}
        for item in anchor_record.get("anchors", [])
    ]
    return sorted(anchors, key=lambda item: (item["idea_id"], item["mention_id"]))


def anchor_set_sha256(anchor_record: dict[str, Any]) -> str:
    return anchor_record["anchor_set_sha256"]


def candidate_edge_id(mention_id: str, idea_id: str, relation: str) -> str:
    digest = canonical_sha256([mention_id, idea_id, relation])[:16]
    return f"edge_{digest}"


def build_relation_candidate_record(
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    proposed_edges: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    engine_version: str,
    prompt_version: str,
    turns_by_id: dict[str, str],
) -> dict[str, Any]:
    """Build and structurally validate a distinct, optional relation-edge artifact."""
    validate_layered_outcome(outcome, turns_by_id)
    if not isinstance(proposed_edges, list):
        raise ValueError("proposed_edges must be a list")
    validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
    anchors = direct_expression_anchors(anchor_record)
    normalized_edges = []
    for edge in proposed_edges:
        if not isinstance(edge, dict) or set(edge) != {"mention_id", "idea_id", "relation"}:
            raise ValueError("each proposed edge must contain exactly mention_id, idea_id, and relation")
        mention_id, idea_id, relation = edge.get("mention_id"), edge.get("idea_id"), edge.get("relation")
        normalized_edges.append({
            "edge_id": candidate_edge_id(mention_id, idea_id, relation)
            if isinstance(mention_id, str) and isinstance(idea_id, str) and isinstance(relation, str)
            else "",
            "mention_id": mention_id,
            "idea_id": idea_id,
            "relation": relation,
        })
    normalized_edges.sort(key=lambda item: item["edge_id"])
    record = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "event_id": outcome.get("event_id"),
        "source_outcome_sha256": canonical_sha256(outcome),
        "source_record_sha256": outcome.get("source_record_sha256"),
        "target_transcript_sha256": outcome.get("target_transcript_sha256"),
        "source_anchor_artifact_sha256": anchor_canonical_sha256(anchor_record),
        "anchor_set_sha256": anchor_set_sha256(anchor_record),
        "direct_expression_anchors": anchors,
        "unanchored_idea_ids": anchor_record["unanchored_idea_ids"],
        "extractor": {
            "provider": provider,
            "model": model,
            "engine_version": engine_version,
            "prompt_version": prompt_version,
        },
        "edges": normalized_edges,
        "validation": {
            "structural_valid": True,
            "direct_expression_anchors_bound_to_source": True,
            "semantic_relation_validated": False,
            "edge_completeness_validated": False,
        },
    }
    return validate_relation_candidate_record(record, outcome, anchor_record, turns_by_id)


def validate_relation_candidate_record(
    record: dict[str, Any],
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    turns_by_id: dict[str, str],
    *,
    max_edges: int = 1200,
) -> dict[str, Any]:
    """Validate IDs, source hashes, anchor projection, and relation-edge structure."""
    if not isinstance(record, dict):
        raise ValueError("relation candidate record must be an object")
    if set(record) != _RECORD_FIELDS:
        raise ValueError(f"relation candidate record must contain exactly: {sorted(_RECORD_FIELDS)}")
    if record.get("schema_version") != 1 or record.get("artifact_type") != _ARTIFACT_TYPE:
        raise ValueError("unsupported relation candidate schema")
    if record.get("event_id") != outcome.get("event_id"):
        raise ValueError("relation candidate event_id does not match its source outcome")
    if record.get("source_outcome_sha256") != canonical_sha256(outcome):
        raise ValueError("relation candidate source outcome hash mismatch")
    for field in ("source_outcome_sha256", "source_record_sha256", "target_transcript_sha256", "source_anchor_artifact_sha256", "anchor_set_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"relation candidate {field} must be a lowercase SHA-256 digest")
    for field in ("source_record_sha256", "target_transcript_sha256"):
        if record.get(field) != outcome.get(field):
            raise ValueError(f"relation candidate {field} does not match its source outcome")
    validate_layered_outcome(outcome, turns_by_id)

    validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
    if record.get("source_anchor_artifact_sha256") != anchor_canonical_sha256(anchor_record):
        raise ValueError("source anchor artifact hash mismatch")
    expected_anchors = direct_expression_anchors(anchor_record)
    anchors = record.get("direct_expression_anchors")
    if anchors != expected_anchors:
        raise ValueError("direct_expression_anchors must exactly project the separate anchor artifact")
    if record.get("unanchored_idea_ids") != anchor_record["unanchored_idea_ids"]:
        raise ValueError("unanchored idea IDs must match the separate anchor artifact")
    if record.get("anchor_set_sha256") != anchor_set_sha256(anchor_record):
        raise ValueError("anchor-set hash mismatch")
    extractor = record.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("extractor provenance must be an object")
    allowed_extractor_fields = {"provider", "model", "engine_version", "prompt_version", "response_sha256"}
    if set(extractor) - allowed_extractor_fields:
        raise ValueError("extractor provenance contains unsupported fields")
    for field in ("provider", "model", "engine_version", "prompt_version"):
        if not isinstance(extractor.get(field), str) or not extractor[field].strip():
            raise ValueError(f"extractor.{field} must be nonempty")
    if "response_sha256" in extractor and (
        not isinstance(extractor["response_sha256"], str) or not _HASH.fullmatch(extractor["response_sha256"])
    ):
        raise ValueError("extractor.response_sha256 must be a lowercase SHA-256 digest")

    edges = record.get("edges")
    if not isinstance(edges, list) or len(edges) > max_edges:
        raise ValueError(f"edges must be a list of at most {max_edges}")
    mention_ids = {mention["mention_id"] for mention in outcome["mentions"]}
    idea_ids = {idea["idea_id"] for idea in outcome["ideas"]}
    anchored_idea_ids = {anchor["idea_id"] for anchor in expected_anchors}
    anchor_pairs = {(anchor["mention_id"], anchor["idea_id"]) for anchor in expected_anchors}
    edge_ids: set[str] = set()
    endpoint_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        field = f"edges[{index}]"
        if not isinstance(edge, dict) or set(edge) != {"edge_id", "mention_id", "idea_id", "relation"}:
            raise ValueError(f"{field} must contain exactly edge_id, mention_id, idea_id, relation")
        edge_id, mention_id, idea_id, relation = (
            edge["edge_id"], edge["mention_id"], edge["idea_id"], edge["relation"]
        )
        if not isinstance(mention_id, str) or mention_id not in mention_ids:
            raise ValueError(f"{field}.mention_id does not resolve in the source outcome")
        if not isinstance(idea_id, str) or idea_id not in idea_ids or idea_id not in anchored_idea_ids:
            raise ValueError(f"{field}.idea_id must resolve to an anchored idea")
        if not isinstance(relation, str) or relation not in FINE_RELATIONS:
            raise ValueError(f"{field}.relation must be one of {sorted(FINE_RELATIONS)}; expresses is an anchor, not a fine edge")
        pair = (mention_id, idea_id)
        if pair in anchor_pairs:
            raise ValueError(f"{field} duplicates an existing direct-expression anchor")
        if pair in endpoint_pairs:
            raise ValueError(f"{field} assigns multiple fine relations to the same mention-idea pair")
        endpoint_pairs.add(pair)
        expected_id = candidate_edge_id(mention_id, idea_id, relation)
        if not isinstance(edge_id, str) or not _EDGE_ID.fullmatch(edge_id) or edge_id != expected_id:
            raise ValueError(f"{field}.edge_id must be the deterministic ID for its endpoints and relation")
        if edge_id in edge_ids:
            raise ValueError(f"{field}.edge_id is duplicated")
        edge_ids.add(edge_id)

    expected_validation = {
        "structural_valid": True,
        "direct_expression_anchors_bound_to_source": True,
        "semantic_relation_validated": False,
        "edge_completeness_validated": False,
    }
    if record.get("validation") != expected_validation:
        raise ValueError("validation flags must distinguish structural from semantic validation")
    return record
