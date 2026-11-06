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
FINE_RELATION_TAXONOMY_VERSION = "fine-relation-taxonomy-v2"
FINE_RELATION_DEFINITIONS = {
    "supports": "Gives a reason, premise, or evidence for the idea; the connection can be paraphrased as ‘this is a reason to believe/act on the idea.’ Use for general evidence, not a concrete case presented as an example.",
    "illustrates": "Presents a concrete case as an instance of the idea; the connection can be paraphrased as ‘here is the idea happening in a particular case.’ Prefer this to supports when the case is framed as an example.",
    "elaborates": "Adds non-limiting scope, mechanism, or implementation detail to the idea; the connection can be paraphrased as ‘more specifically, the idea means/includes this.’ It is not mainly evidence or a concrete instance.",
    "qualifies": "Changes the idea’s scope or truth conditions by limiting, conditioning, correcting, or stating an exception. Mere additional specification is not enough.",
    "contextualizes": "Supplies relevant setting/background that aids interpretation but does not itself give a reason, instantiate the idea, specify its content, or alter its scope.",
}
FINE_RELATION_DECISION_RULES = [
    "If the mention itself directly states the idea, represent that only as an expresses anchor; do not emit a fine edge for that pair.",
    "Choose qualifies only when the mention restricts, conditions, corrects, or excepts the idea; a non-limiting implementation detail is elaborates.",
    "Choose illustrates for a particular case explicitly serving as an instance; choose supports for general evidence or a reason. If that distinction is not clear, abstain.",
    "Choose elaborates for general scope, mechanism, or implementation detail that changes what the idea includes without limiting/correcting it.",
    "Choose contextualizes only for relevant background with no clear evidential, illustrative, specification, or limiting relation.",
    "Emit at most one fine relation per mention/idea pair. If multiple labels remain plausible or no relation is clear, omit the edge.",
]
FINE_RELATION_BOUNDARY_EXAMPLES = [
    {
        "idea": "Extend evening bus service.",
        "mention": "The last bus leaves at 9:45, although many hospital shifts end after 10.",
        "relation": "supports",
        "why": "General evidence/reason for extending service, not a particular case used as an example.",
    },
    {
        "idea": "Extend evening bus service.",
        "mention": "A nurse leaving the hospital at 10:30 has no bus and walks four miles home.",
        "relation": "illustrates",
        "why": "A particular case is presented as an instance of the service gap; do not choose supports merely because it is also persuasive.",
    },
    {
        "idea": "Extend evening bus service.",
        "mention": "The proposal would run hourly buses on routes 2 and 6 until midnight.",
        "relation": "elaborates",
        "why": "Adds general route and schedule details without narrowing or correcting a stated claim.",
    },
    {
        "idea": "Extend evening bus service.",
        "mention": "The extension would apply only on weekends.",
        "relation": "qualifies",
        "why": "The explicit ‘only’ limits the scope of the general proposal; absent that limitation, extra schedule detail would elaborate.",
    },
    {
        "idea": "Extend evening bus service.",
        "mention": "The city added two hospital districts as development spread westward.",
        "relation": "contextualizes",
        "why": "Relevant setting only; if the mention explicitly argues that growth makes evening service necessary, it would instead support the idea.",
    },
]
_EDGE_ID = re.compile(r"^edge_[0-9a-f]{16}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE = "layered-idea-relation-candidates"
_ERROR_ARTIFACT_TYPE = "layered-idea-relation-candidate-error"
_ERROR_RECORD_FIELDS = {
    "schema_version", "artifact_type", "status", "event_id", "source_outcome_sha256",
    "source_record_sha256", "target_transcript_sha256", "source_anchor_artifact_sha256",
    "anchor_set_sha256", "extractor", "attempt_count", "errors",
}
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


def validate_proposed_relation_edges(
    proposed_edges: Any,
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    *,
    max_edges: int = 1200,
) -> list[dict[str, str]]:
    """Reject invalid model edges before constructing or serializing a candidate record."""
    if not isinstance(proposed_edges, list) or len(proposed_edges) > max_edges:
        raise ValueError(f"proposed_edges must be a list of at most {max_edges}")
    mention_ids = {mention["mention_id"] for mention in outcome.get("mentions", []) if isinstance(mention, dict)}
    idea_ids = {idea["idea_id"] for idea in outcome.get("ideas", []) if isinstance(idea, dict)}
    anchors = direct_expression_anchors(anchor_record)
    anchored_idea_ids = {anchor["idea_id"] for anchor in anchors}
    anchor_pairs = {(anchor["mention_id"], anchor["idea_id"]) for anchor in anchors}
    endpoint_pairs: set[tuple[str, str]] = set()
    checked = []
    for index, edge in enumerate(proposed_edges):
        field = f"proposed_edges[{index}]"
        if not isinstance(edge, dict) or set(edge) != {"mention_id", "idea_id", "relation"}:
            raise ValueError(f"{field} must contain exactly mention_id, idea_id, and relation")
        mention_id, idea_id, relation = edge["mention_id"], edge["idea_id"], edge["relation"]
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
        checked.append({"mention_id": mention_id, "idea_id": idea_id, "relation": relation})
    return checked


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
    taxonomy_version: str = FINE_RELATION_TAXONOMY_VERSION,
    attempt_count: int = 1,
    repair_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build and structurally validate a distinct, optional relation-edge artifact."""
    validate_layered_outcome(outcome, turns_by_id)
    validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
    checked_edges = validate_proposed_relation_edges(proposed_edges, outcome, anchor_record)
    anchors = direct_expression_anchors(anchor_record)
    normalized_edges = [{
        "edge_id": candidate_edge_id(edge["mention_id"], edge["idea_id"], edge["relation"]),
        **edge,
    } for edge in checked_edges]
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
            "taxonomy_version": taxonomy_version,
            "attempt_count": attempt_count,
            "repair_errors": repair_errors or [],
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


def build_relation_error_record(
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    errors: list[str],
    *,
    attempt_count: int,
    provider: str,
    model: str,
    engine_version: str,
    prompt_version: str,
    turns_by_id: dict[str, str],
    response_sha256: str | None = None,
    taxonomy_version: str = FINE_RELATION_TAXONOMY_VERSION,
) -> dict[str, Any]:
    """Build an explicit failed-event row; failure is not a zero-edge result."""
    validate_layered_outcome(outcome, turns_by_id)
    validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
    record = {
        "schema_version": 1,
        "artifact_type": _ERROR_ARTIFACT_TYPE,
        "status": "model_error",
        "event_id": outcome.get("event_id"),
        "source_outcome_sha256": canonical_sha256(outcome),
        "source_record_sha256": outcome.get("source_record_sha256"),
        "target_transcript_sha256": outcome.get("target_transcript_sha256"),
        "source_anchor_artifact_sha256": anchor_canonical_sha256(anchor_record),
        "anchor_set_sha256": anchor_set_sha256(anchor_record),
        "extractor": {
            "provider": provider,
            "model": model,
            "engine_version": engine_version,
            "prompt_version": prompt_version,
            "taxonomy_version": taxonomy_version,
        },
        "attempt_count": attempt_count,
        "errors": errors,
    }
    if response_sha256 is not None:
        record["extractor"]["response_sha256"] = response_sha256
    return validate_relation_error_record(record, outcome, anchor_record, turns_by_id)


def validate_relation_error_record(
    record: dict[str, Any],
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    turns_by_id: dict[str, str],
) -> dict[str, Any]:
    """Validate the provenance and explicit failure status of a relation error row."""
    if not isinstance(record, dict) or set(record) != _ERROR_RECORD_FIELDS:
        raise ValueError(f"relation error record must contain exactly: {sorted(_ERROR_RECORD_FIELDS)}")
    if record.get("schema_version") != 1 or record.get("artifact_type") != _ERROR_ARTIFACT_TYPE:
        raise ValueError("unsupported relation error schema")
    if record.get("status") != "model_error":
        raise ValueError("relation error record must have status=model_error")
    if record.get("event_id") != outcome.get("event_id"):
        raise ValueError("relation error event_id does not match its source outcome")
    if record.get("source_outcome_sha256") != canonical_sha256(outcome):
        raise ValueError("relation error source outcome hash mismatch")
    for field in ("source_outcome_sha256", "source_record_sha256", "target_transcript_sha256", "source_anchor_artifact_sha256", "anchor_set_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"relation error {field} must be a lowercase SHA-256 digest")
    for field in ("source_record_sha256", "target_transcript_sha256"):
        if record.get(field) != outcome.get(field):
            raise ValueError(f"relation error {field} does not match its source outcome")
    validate_layered_outcome(outcome, turns_by_id)
    validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
    if record.get("source_anchor_artifact_sha256") != anchor_canonical_sha256(anchor_record):
        raise ValueError("relation error source anchor artifact hash mismatch")
    if record.get("anchor_set_sha256") != anchor_set_sha256(anchor_record):
        raise ValueError("relation error anchor-set hash mismatch")
    extractor = record.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("extractor provenance must be an object")
    allowed = {"provider", "model", "engine_version", "prompt_version", "taxonomy_version", "response_sha256"}
    if set(extractor) - allowed:
        raise ValueError("extractor provenance contains unsupported fields")
    for field in ("provider", "model", "engine_version", "prompt_version", "taxonomy_version"):
        if not isinstance(extractor.get(field), str) or not extractor[field].strip():
            raise ValueError(f"extractor.{field} must be nonempty")
    if "response_sha256" in extractor and (
        not isinstance(extractor["response_sha256"], str) or not _HASH.fullmatch(extractor["response_sha256"])
    ):
        raise ValueError("extractor.response_sha256 must be a lowercase SHA-256 digest")
    count = record.get("attempt_count")
    errors = record.get("errors")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > 3:
        raise ValueError("attempt_count must be an integer from 1 to 3")
    if not isinstance(errors, list) or len(errors) != count or any(not isinstance(error, str) or not error.strip() for error in errors):
        raise ValueError("errors must contain one nonempty message per failed attempt")
    return record


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
    allowed_extractor_fields = {
        "provider", "model", "engine_version", "prompt_version", "taxonomy_version", "response_sha256",
        "attempt_count", "repair_errors",
    }
    if set(extractor) - allowed_extractor_fields:
        raise ValueError("extractor provenance contains unsupported fields")
    for field in ("provider", "model", "engine_version", "prompt_version"):
        if not isinstance(extractor.get(field), str) or not extractor[field].strip():
            raise ValueError(f"extractor.{field} must be nonempty")
    if "response_sha256" in extractor and (
        not isinstance(extractor["response_sha256"], str) or not _HASH.fullmatch(extractor["response_sha256"])
    ):
        raise ValueError("extractor.response_sha256 must be a lowercase SHA-256 digest")
    if "taxonomy_version" in extractor and (
        not isinstance(extractor["taxonomy_version"], str) or not extractor["taxonomy_version"].strip()
    ):
        raise ValueError("extractor.taxonomy_version must be nonempty")
    attempt_count = extractor.get("attempt_count", 1)
    repair_errors = extractor.get("repair_errors", [])
    if not isinstance(attempt_count, int) or isinstance(attempt_count, bool) or not 1 <= attempt_count <= 3:
        raise ValueError("extractor.attempt_count must be an integer from 1 to 3")
    if (
        not isinstance(repair_errors, list)
        or len(repair_errors) != attempt_count - 1
        or any(not isinstance(error, str) or not error.strip() for error in repair_errors)
    ):
        raise ValueError("extractor.repair_errors must contain one nonempty message per repaired attempt")

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
