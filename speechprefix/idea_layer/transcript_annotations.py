"""Occurrence-based transcript outcomes; legacy v1/v2 outcomes are unchanged."""
from __future__ import annotations

import re

_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
LAYERS = {
    "mentions": "mention_id", "ideas": "idea_id", "key_points": "key_point_id",
    "topics": "topic_id", "opinions": "opinion_id", "emotions": "emotion_id",
}
EDGES = {
    "idea_mentions": ("ideas", "mentions"),
    "key_point_ideas": ("key_points", "ideas"),
    "key_point_mentions": ("key_points", "mentions"),
    "topic_ideas": ("topics", "ideas"),
    "opinion_mentions": ("opinions", "mentions"),
    "emotion_mentions": ("emotions", "mentions"),
}
STANCES = ("asserts", "supports", "opposes", "proposes", "requests", "forecasts", "unclear")


class TranscriptAnnotationError(ValueError):
    """Invalid annotation structure or transcript provenance."""


def validate_transcript_annotation(record: dict, turns_by_id: dict[str, str]) -> dict:
    """Validate typed links and exact target-turn quotes, not semantic coverage."""
    def require(condition, message):
        if not condition:
            raise TranscriptAnnotationError(message)

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    require(isinstance(record, dict) and record.get("artifact_type") == "transcript-annotation"
            and type(record.get("schema_version")) is int and record["schema_version"] == 1,
            "unsupported transcript annotation schema")
    require(text(record.get("event_id")), "event_id must be nonempty")
    ids = {layer: set() for layer in LAYERS}
    for layer, id_field in LAYERS.items():
        rows = record.get(layer)
        require(isinstance(rows, list), f"{layer} must be a list")
        for row in rows:
            require(isinstance(row, dict), f"{layer} node must be an object")
            ident = row.get(id_field)
            require(isinstance(ident, str) and _ID.fullmatch(ident) and ident not in ids[layer],
                    f"{id_field} must be unique snake_case")
            ids[layer].add(ident)
            if layer == "mentions":
                evidence = row.get("evidence")
                require(isinstance(evidence, list) and evidence, f"{ident}: evidence required")
                for cite in evidence:
                    turn_id = cite.get("turn_id") if isinstance(cite, dict) else None
                    require(isinstance(turn_id, str) and turn_id in turns_by_id,
                            f"{ident}: unknown target-speaker turn")
                    quote = cite.get("exact_quote")
                    require(text(quote) and quote in turns_by_id[turn_id],
                            f"{ident}: quote is not an exact transcript substring")
            else:
                field = {"ideas": "occurrence", "key_points": "detail", "topics": "label",
                         "opinions": "text", "emotions": "label"}[layer]
                require(text(row.get(field)), f"{ident}: {field} required")
            if layer == "opinions":
                require(row.get("attribution") == "speaker" and row.get("stance") in STANCES,
                        f"{ident}: speaker attribution and stance required")
            if layer in ("opinions", "emotions") or "uncertainty" in row:
                require(text(row.get("uncertainty")), f"{ident}: uncertainty required")
    linked_mentions = set()
    grouped_ideas = set()
    for relation, (left, right) in EDGES.items():
        edges = record.get(relation)
        require(isinstance(edges, list), f"{relation} must be a list")
        seen = set()
        for edge in edges:
            require(isinstance(edge, dict), f"{relation} edge must be an object")
            a, b = edge.get(LAYERS[left]), edge.get(LAYERS[right])
            require(isinstance(a, str) and a in ids[left] and isinstance(b, str) and b in ids[right],
                    f"{relation} has unresolved endpoint")
            require((a, b) not in seen, f"duplicate {relation} edge")
            seen.add((a, b))
            if relation == "idea_mentions":
                linked_mentions.add(b)
            if relation == "topic_ideas":
                grouped_ideas.add(b)
        require({a for a, _ in seen} == ids[left], f"unlinked {left} in {relation}")
    require(linked_mentions == ids["mentions"], "mentions must link to ideas")
    require(grouped_ideas == ids["ideas"], "ideas must link to topics")
    require(len({e['key_point_id'] for e in record['key_point_ideas']}) == len(record['key_point_ideas']),
            "each key point must have one parent idea")
    edges = record.get("opinion_targets")
    require(isinstance(edges, list), "opinion_targets must be a list")
    targets = set()
    for edge in edges:
        require(isinstance(edge, dict), "opinion_targets edge must be an object")
        a, b, kind = edge.get("opinion_id"), edge.get("target_id"), edge.get("target_type")
        require(isinstance(a, str) and a in ids["opinions"], "unresolved opinion_id")
        require(kind in ("idea", "key_point") and isinstance(b, str) and b in ids[kind + "s"],
                "unresolved target_id or target_type")
        require((a, b, kind) not in targets, "duplicate opinion target")
        targets.add((a, b, kind))
    require({a for a, _, _ in targets} == ids["opinions"], "opinions must have targets")
    require(record.get("coverage") == "full_pass", "full_pass coverage attestation required")
    if not ids["ideas"]:
        require(text(record.get("empty_reason")), "empty inventory requires a reason")
    return {**record, "artifact_role": "outcomes", "validation": {
        "structural_valid": True, "exact_quote_integrity": True,
        "semantic_entailment_validated": False, "inventory_completeness_validated": False,
    }}
