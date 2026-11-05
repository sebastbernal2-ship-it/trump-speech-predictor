"""Provenance- and time-aware heterogeneous news/story graph contracts.

This module defines a graph-ready data boundary; it intentionally does not
cluster stories or assign relevance weights. Those algorithms must be
separately versioned and evaluated before they can feed a forecasting model.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from typing import Any

from .contracts import is_source_eligible_at, validate_source_evidence


NODE_TYPES = frozenset({
    "article_observation", "publisher", "story", "entity", "entity_claim",
    "concept", "target_event", "idea_candidate",
})
EDGE_TYPES = frozenset({
    "member_of", "published_by", "owned_by", "exact_duplicate_of",
    "syndicated_copy_of", "updates", "related_to", "mentions", "claim_about",
    "associated_with", "employed_by", "met_with", "endorses", "opposes",
    "located_in", "candidate_evidence",
})
AVAILABILITY_STATES = frozenset({"verified_before_cutoff", "not_verified", "unknown"})
ALGORITHM_FIT_MODES = frozenset({"deterministic", "learned"})
_FORBIDDEN_KEYS = frozenset({
    "transcript", "transcript_text", "target_text", "target_ideas", "theme_labels",
    "draft_themes", "outcome_labels", "post_event_notes", "future_turns", "target_summary",
    "realized_idea", "realized_target", "target_idea", "target_label", "target_labels",
    "target_outcome", "matched_target", "target_match", "transcript_ideas", "gold_label", "gold_labels",
})
_GRAPH_KEYS = {"schema_version", "record_type", "nodes", "edges", "metadata", "graph_sha256"}
_NODE_REQUIRED = {"node_id", "node_type", "source_ids", "availability_status", "attributes"}
_NODE_OPTIONAL = {"available_at", "published_at", "valid_from", "valid_until", "confidence"}
_EDGE_REQUIRED = {
    "edge_id", "source_node_id", "target_node_id", "relation", "source_ids",
    "availability_status", "confidence",
}
_EDGE_OPTIONAL = {"available_at", "published_at", "valid_from", "valid_until", "attributes"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _timestamp(value: Any, field: str, *, allow_date: bool = False) -> datetime:
    if allow_date and isinstance(value, str):
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            pass
        else:
            return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    if not isinstance(value, str) or "T" not in value:
        expected = "an ISO date or timezone-aware RFC 3339 timestamp" if allow_date else "a timezone-aware RFC 3339 timestamp"
        raise ValueError(f"{field} must be {expected}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be a timezone-aware RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _reject_outcomes(value: Any, field: str) -> None:
    if isinstance(value, dict):
        keys = {str(key).casefold().replace("-", "_").replace(" ", "_") for key in value}
        forbidden = _FORBIDDEN_KEYS.intersection(keys)
        if forbidden:
            raise ValueError(f"{field} contains outcome-only fields: {sorted(forbidden)}")
        for key, nested in value.items():
            _reject_outcomes(nested, f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_outcomes(nested, f"{field}[{index}]")


def seal_story_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible copy sealed with a canonical SHA-256 digest."""
    sealed = json.loads(json.dumps(graph, ensure_ascii=False, allow_nan=False))
    sealed.pop("graph_sha256", None)
    sealed["graph_sha256"] = _digest(sealed)
    return sealed


def _validate_temporal_fields(record: dict[str, Any], field: str) -> None:
    status = record.get("availability_status")
    if not isinstance(status, str) or status not in AVAILABILITY_STATES:
        raise ValueError(f"{field}.availability_status is unsupported")
    if "available_at" in record:
        _timestamp(record["available_at"], f"{field}.available_at")
    if "published_at" in record:
        _timestamp(record["published_at"], f"{field}.published_at", allow_date=True)
    if status == "verified_before_cutoff" and "available_at" not in record:
        raise ValueError(f"{field} needs available_at when availability is verified")
    if "valid_from" in record:
        _timestamp(record["valid_from"], f"{field}.valid_from", allow_date=True)
    if "valid_until" in record:
        _timestamp(record["valid_until"], f"{field}.valid_until", allow_date=True)
    if "valid_from" in record and "valid_until" in record:
        if _timestamp(record["valid_from"], f"{field}.valid_from", allow_date=True) >= _timestamp(record["valid_until"], f"{field}.valid_until", allow_date=True):
            raise ValueError(f"{field} valid_from must precede valid_until")


def validate_story_graph(
    graph: dict[str, Any], source_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate graph schema, typed links, temporal provenance, and content hash.

    When an evidence ledger is supplied, every node/edge citation is resolved
    and validated against its immutable source record.
    """
    if not isinstance(graph, dict) or set(graph) != _GRAPH_KEYS:
        raise ValueError(f"story graph must contain exactly: {sorted(_GRAPH_KEYS)}")
    if graph.get("schema_version") != 1 or graph.get("record_type") != "idea-layer-story-graph":
        raise ValueError("story graph must use schema_version 1 and record_type idea-layer-story-graph")
    if not isinstance(graph["metadata"], dict):
        raise ValueError("story graph metadata must be an object")
    if not isinstance(graph["metadata"].get("algorithm_version"), str) or not graph["metadata"]["algorithm_version"].strip():
        raise ValueError("story graph metadata needs a nonempty algorithm_version")
    fit_mode = graph["metadata"].get("algorithm_fit_mode")
    if not isinstance(fit_mode, str) or fit_mode not in ALGORITHM_FIT_MODES:
        raise ValueError("story graph metadata needs algorithm_fit_mode: deterministic or learned")
    if fit_mode == "learned" and "algorithm_training_cutoff_at" not in graph["metadata"]:
        raise ValueError("learned graph algorithms need algorithm_training_cutoff_at")
    if "algorithm_training_cutoff_at" in graph["metadata"]:
        _timestamp(graph["metadata"]["algorithm_training_cutoff_at"], "metadata.algorithm_training_cutoff_at")
    evidence_digest = graph["metadata"].get("source_evidence_sha256")
    if evidence_digest is not None and (
        not isinstance(evidence_digest, str) or len(evidence_digest) != 64
        or any(char not in "0123456789abcdef" for char in evidence_digest)
    ):
        raise ValueError("metadata.source_evidence_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        raise ValueError("story graph nodes and edges must be lists")
    _reject_outcomes(graph, "story_graph")

    node_by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(graph["nodes"]):
        field = f"nodes[{index}]"
        if not isinstance(node, dict) or not _NODE_REQUIRED <= set(node) or set(node) - (_NODE_REQUIRED | _NODE_OPTIONAL):
            raise ValueError(f"{field} has missing or unsupported fields")
        node_id = _text(node["node_id"], f"{field}.node_id")
        if node_id in node_by_id:
            raise ValueError(f"duplicate graph node ID: {node_id}")
        if not isinstance(node["node_type"], str) or node["node_type"] not in NODE_TYPES:
            raise ValueError(f"{field}.node_type is unsupported: {node['node_type']!r}")
        if not isinstance(node["source_ids"], list) or not node["source_ids"]:
            raise ValueError(f"{field}.source_ids must be a nonempty list")
        if any(not isinstance(source_id, str) or not source_id.strip() for source_id in node["source_ids"]):
            raise ValueError(f"{field}.source_ids must contain nonempty strings")
        if len(set(node["source_ids"])) != len(node["source_ids"]):
            raise ValueError(f"{field}.source_ids must be unique")
        if not isinstance(node["attributes"], dict):
            raise ValueError(f"{field}.attributes must be an object")
        _validate_temporal_fields(node, field)
        if "confidence" in node and (
            isinstance(node["confidence"], bool) or not isinstance(node["confidence"], (int, float))
            or not math.isfinite(node["confidence"]) or not 0.0 <= node["confidence"] <= 1.0
        ):
            raise ValueError(f"{field}.confidence must be finite and between 0 and 1")
        node_by_id[node_id] = node

    edge_ids = set()
    for index, edge in enumerate(graph["edges"]):
        field = f"edges[{index}]"
        if not isinstance(edge, dict) or not _EDGE_REQUIRED <= set(edge) or set(edge) - (_EDGE_REQUIRED | _EDGE_OPTIONAL):
            raise ValueError(f"{field} has missing or unsupported fields")
        edge_id = _text(edge["edge_id"], f"{field}.edge_id")
        if edge_id in edge_ids:
            raise ValueError(f"duplicate graph edge ID: {edge_id}")
        edge_ids.add(edge_id)
        source_node_id = _text(edge["source_node_id"], f"{field}.source_node_id")
        target_node_id = _text(edge["target_node_id"], f"{field}.target_node_id")
        if source_node_id not in node_by_id or target_node_id not in node_by_id:
            raise ValueError(f"{field} references a missing endpoint")
        endpoints = (node_by_id[source_node_id], node_by_id[target_node_id])
        if edge["availability_status"] == "verified_before_cutoff":
            if any(endpoint["availability_status"] != "verified_before_cutoff" for endpoint in endpoints):
                raise ValueError(f"{field} cannot be verified before cutoff when an endpoint is unverified")
        if "available_at" in edge:
            edge_available = _timestamp(edge["available_at"], f"{field}.available_at")
            if any(
                "available_at" in endpoint
                and _timestamp(endpoint["available_at"], "endpoint.available_at") > edge_available
                for endpoint in endpoints
            ):
                raise ValueError(f"{field}.available_at cannot precede endpoint availability")
        if not isinstance(edge["relation"], str) or edge["relation"] not in EDGE_TYPES:
            raise ValueError(f"{field}.relation is unsupported: {edge['relation']!r}")
        if not isinstance(edge["source_ids"], list) or not edge["source_ids"]:
            raise ValueError(f"{field}.source_ids must be a nonempty list")
        if any(not isinstance(source_id, str) or not source_id.strip() for source_id in edge["source_ids"]):
            raise ValueError(f"{field}.source_ids must contain nonempty strings")
        if len(set(edge["source_ids"])) != len(edge["source_ids"]):
            raise ValueError(f"{field}.source_ids must be unique")
        if "attributes" in edge and not isinstance(edge["attributes"], dict):
            raise ValueError(f"{field}.attributes must be an object")
        _validate_temporal_fields(edge, field)
        confidence = edge["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{field}.confidence must be finite and between 0 and 1")

    if source_evidence is not None:
        source_by_id = {}
        for source in source_evidence:
            source_id = validate_source_evidence(source)["source_id"]
            if source_id in source_by_id:
                raise ValueError(f"duplicate source evidence ID: {source_id}")
            source_by_id[source_id] = source
        referenced_ids = {
            source_id
            for record in [*graph["nodes"], *graph["edges"]]
            for source_id in record["source_ids"]
        }
        missing_source_ids = sorted(referenced_ids - set(source_by_id))
        if missing_source_ids:
            raise ValueError(f"story graph references missing source evidence: {missing_source_ids}")
        evidence_digest = _digest([source_by_id[source_id] for source_id in sorted(referenced_ids)])
        recorded_evidence_digest = graph["metadata"].get("source_evidence_sha256")
        if recorded_evidence_digest is not None and recorded_evidence_digest != evidence_digest:
            raise ValueError("source_evidence_sha256 does not match supplied source evidence")
        for record in [*graph["nodes"], *graph["edges"]]:
            if record["availability_status"] != "verified_before_cutoff":
                continue
            cited = [source_by_id[source_id] for source_id in record["source_ids"]]
            if any(source["availability_status"] != "verified_before_cutoff" for source in cited):
                raise ValueError("verified graph records may cite only verified source evidence")
            latest_observation = max(_timestamp(source["observed_at"], "source.observed_at") for source in cited)
            if _timestamp(record["available_at"], "record.available_at") < latest_observation:
                raise ValueError("verified graph record availability cannot precede cited source observation")

    supplied_digest = graph.get("graph_sha256")
    if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
        raise ValueError("graph_sha256 must be a lowercase SHA-256 digest")
    if any(char not in "0123456789abcdef" for char in supplied_digest):
        raise ValueError("graph_sha256 must be a lowercase SHA-256 digest")
    unhashed = {key: value for key, value in graph.items() if key != "graph_sha256"}
    if _digest(unhashed) != supplied_digest:
        raise ValueError("graph_sha256 does not match graph content")
    return {"node_count": len(node_by_id), "edge_count": len(edge_ids), "graph_sha256": supplied_digest}


def story_graph_as_of(
    graph: dict[str, Any], cutoff_at: str, *, source_evidence: list[dict[str, Any]],
    availability_mode: str = "strict",
) -> dict[str, Any]:
    """Return a sealed subgraph containing only evidence usable before a cutoff.

    ``strict`` admits only records explicitly verified available before the
    cutoff. ``retrospective`` also admits records with a pre-cutoff publication
    date, but labels the result as not point-in-time verified.
    """
    validate_story_graph(graph, source_evidence)
    cutoff = _timestamp(cutoff_at, "cutoff_at")
    source_by_id = {row["source_id"]: row for row in source_evidence}
    if availability_mode not in {"strict", "retrospective"}:
        raise ValueError("availability_mode must be 'strict' or 'retrospective'")
    if availability_mode == "strict" and graph["metadata"]["algorithm_fit_mode"] == "learned":
        training_cutoff = _timestamp(
            graph["metadata"]["algorithm_training_cutoff_at"], "algorithm_training_cutoff_at",
        )
        if training_cutoff >= cutoff:
            raise ValueError("learned graph algorithm training cutoff must precede event cutoff")

    if availability_mode == "strict":
        eligible_source_ids = {
            source_id for source_id, source in source_by_id.items()
            if is_source_eligible_at(source, cutoff.isoformat())
        }
    else:
        eligible_source_ids = {
            source_id for source_id, source in source_by_id.items()
            if _timestamp(source["published_at"], "source.published_at") < cutoff
        }

    def eligible(record: dict[str, Any]) -> bool:
        if not set(record["source_ids"]) <= eligible_source_ids:
            return False
        status = record["availability_status"]
        if availability_mode == "strict":
            if status != "verified_before_cutoff" or "available_at" not in record:
                return False
            available = _timestamp(record["available_at"], "available_at")
        else:
            if status == "verified_before_cutoff" and "available_at" in record:
                available = _timestamp(record["available_at"], "available_at")
            elif "published_at" in record:
                published = record["published_at"]
                if isinstance(published, str) and "T" not in published:
                    try:
                        if date.fromisoformat(published) >= cutoff.date():
                            return False
                    except ValueError:
                        return False
                available = _timestamp(published, "published_at", allow_date=True)
            else:
                return False
        if available >= cutoff:
            return False
        valid_from = _timestamp(record["valid_from"], "valid_from", allow_date=True) if "valid_from" in record else None
        valid_until = _timestamp(record["valid_until"], "valid_until", allow_date=True) if "valid_until" in record else None
        return (valid_from is None or valid_from <= cutoff) and (valid_until is None or cutoff < valid_until)

    nodes = [node for node in graph["nodes"] if eligible(node)]
    node_ids = {node["node_id"] for node in nodes}
    edges = [
        edge for edge in graph["edges"]
        if edge["source_node_id"] in node_ids and edge["target_node_id"] in node_ids and eligible(edge)
    ]
    used_source_ids = sorted({
        source_id for record in [*nodes, *edges] for source_id in record["source_ids"]
    })
    used_source_digest = _digest([source_by_id[source_id] for source_id in used_source_ids])
    result = {
        "schema_version": 1,
        "record_type": "idea-layer-story-graph",
        "metadata": {
            **graph["metadata"],
            "as_of_cutoff_at": cutoff.isoformat(),
            "availability_mode": availability_mode,
            "point_in_time_verified": availability_mode == "strict",
            "parent_graph_sha256": graph["graph_sha256"],
            "source_evidence_sha256": used_source_digest,
        },
        "nodes": sorted(nodes, key=lambda row: row["node_id"]),
        "edges": sorted(edges, key=lambda row: row["edge_id"]),
    }
    sealed = seal_story_graph(result)
    validate_story_graph(sealed, source_evidence)
    return sealed
