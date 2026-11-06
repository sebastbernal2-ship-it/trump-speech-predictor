"""Contracts for layered transcript outcomes: cited mentions linked to abstract ideas.

Mentions preserve transcript-specific content at a finer granularity. Ideas are
separate forecast-target candidates; links explain how the transcript content
expresses, supports, or qualifies each idea. The contract validates provenance,
not semantic correctness or completeness.
"""

from __future__ import annotations

import re
from typing import Any

_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
MENTION_ROLES = {
    "core_expression",
    "supporting_detail",
    "illustrative_example",
    "event_specific_detail",
    "ceremonial_or_personal",
    "other",
}
SALIENCE_LEVELS = {"central", "repeated", "secondary", "brief"}
IDEA_KINDS = {
    "policy",
    "political",
    "factual_claim",
    "campaign_appeal",
    "ceremonial",
    "personal_anecdote",
    "other",
}
STANCES = {"support", "oppose", "describe", "unclear"}
PROMINENCE_LEVELS = {"central", "repeated", "secondary", "brief"}
LINK_RELATIONS = {"expresses", "supports", "illustrates", "elaborates", "qualifies", "contextualizes"}


class LayeredOutcomeError(ValueError):
    """Raised when a layered transcript outcome violates its data contract."""


def validate_layered_outcome(
    record: dict[str, Any],
    turns_by_id: dict[str, str],
    *,
    max_mentions: int = 500,
    max_ideas: int = 200,
    max_links: int = 1200,
) -> dict[str, Any]:
    """Validate a layered outcome row and its evidence against target-speaker turns.

    This is intentionally a structural/provenance validator. It cannot establish
    that a quote entails a mention, that an abstraction is appropriate, or that
    the inventory covers every meaningful idea in the transcript.
    """
    if not isinstance(record, dict):
        raise LayeredOutcomeError("outcome must be an object")
    schema_version = record.get("schema_version")
    if schema_version not in {1, 2} or record.get("artifact_type") != "layered-transcript-outcome":
        raise LayeredOutcomeError("unsupported layered outcome schema")
    if not isinstance(record.get("event_id"), str) or not record["event_id"]:
        raise LayeredOutcomeError("event_id must be nonempty")
    mentions, ideas, links = record.get("mentions"), record.get("ideas"), record.get("links")
    if not isinstance(mentions, list) or len(mentions) > max_mentions:
        raise LayeredOutcomeError(f"mentions must be a list of at most {max_mentions}")
    if not isinstance(ideas, list) or len(ideas) > max_ideas:
        raise LayeredOutcomeError(f"ideas must be a list of at most {max_ideas}")
    if not isinstance(links, list) or len(links) > max_links:
        raise LayeredOutcomeError(f"links must be a list of at most {max_links}")

    mention_ids: set[str] = set()
    normalized_mentions = []
    evidence_count = 0
    for index, mention in enumerate(mentions):
        field = f"mentions[{index}]"
        if not isinstance(mention, dict):
            raise LayeredOutcomeError(f"{field} must be an object")
        mention_id = mention.get("mention_id")
        if not isinstance(mention_id, str) or not _ID.fullmatch(mention_id) or mention_id in mention_ids:
            raise LayeredOutcomeError(f"{field}.mention_id must be unique lowercase snake_case")
        mention_ids.add(mention_id)
        for name in ("text",):
            if not isinstance(mention.get(name), str) or not mention[name].strip():
                raise LayeredOutcomeError(f"{field}.{name} must be nonempty text")
        role, salience = mention.get("role"), mention.get("salience")
        if role not in MENTION_ROLES:
            raise LayeredOutcomeError(f"{field}.role is unsupported")
        if salience not in SALIENCE_LEVELS:
            raise LayeredOutcomeError(f"{field}.salience is unsupported")
        evidence = mention.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4:
            raise LayeredOutcomeError(f"{field}.evidence must contain 1–4 citations")
        checked_evidence = []
        for evidence_index, citation in enumerate(evidence):
            evidence_field = f"{field}.evidence[{evidence_index}]"
            if not isinstance(citation, dict):
                raise LayeredOutcomeError(f"{evidence_field} must be an object")
            turn_id, quote = citation.get("turn_id"), citation.get("exact_quote")
            if not isinstance(turn_id, str) or turn_id not in turns_by_id:
                raise LayeredOutcomeError(f"{evidence_field} cites an unknown target-speaker turn")
            if not isinstance(quote, str) or not quote.strip() or quote not in turns_by_id[turn_id]:
                raise LayeredOutcomeError(f"{evidence_field} quote is not an exact transcript substring")
            evidence_count += 1
            checked_evidence.append({"turn_id": turn_id, "exact_quote": quote})
        normalized_mentions.append({
            "mention_id": mention_id,
            "text": mention["text"].strip(),
            "role": role,
            "salience": salience,
            "evidence": checked_evidence,
        })

    idea_ids: set[str] = set()
    normalized_ideas = []
    for index, idea in enumerate(ideas):
        field = f"ideas[{index}]"
        if not isinstance(idea, dict):
            raise LayeredOutcomeError(f"{field} must be an object")
        idea_id = idea.get("idea_id")
        if not isinstance(idea_id, str) or not _ID.fullmatch(idea_id) or idea_id in idea_ids:
            raise LayeredOutcomeError(f"{field}.idea_id must be unique lowercase snake_case")
        idea_ids.add(idea_id)
        for name in ("concept", "summary"):
            if not isinstance(idea.get(name), str) or not idea[name].strip():
                raise LayeredOutcomeError(f"{field}.{name} must be nonempty text")
        if idea.get("kind") not in IDEA_KINDS:
            raise LayeredOutcomeError(f"{field}.kind is unsupported")
        if idea.get("stance") not in STANCES:
            raise LayeredOutcomeError(f"{field}.stance is unsupported")
        if idea.get("prominence") not in PROMINENCE_LEVELS:
            raise LayeredOutcomeError(f"{field}.prominence is unsupported")
        if not isinstance(idea.get("forecastable"), bool):
            raise LayeredOutcomeError(f"{field}.forecastable must be boolean")
        normalized_ideas.append({
            "idea_id": idea_id,
            "concept": idea["concept"].strip(),
            "summary": idea["summary"].strip(),
            "kind": idea["kind"],
            "stance": idea["stance"],
            "prominence": idea["prominence"],
            "forecastable": idea["forecastable"],
        })

    normalized_links = []
    unique_links: set[tuple[str, str, str]] = set()
    linked_mentions: set[str] = set()
    linked_ideas: set[str] = set()
    directly_expressed_ideas: set[str] = set()
    for index, link in enumerate(links):
        field = f"links[{index}]"
        if not isinstance(link, dict):
            raise LayeredOutcomeError(f"{field} must be an object")
        mention_id, idea_id, relation = link.get("mention_id"), link.get("idea_id"), link.get("relation")
        if mention_id not in mention_ids:
            raise LayeredOutcomeError(f"{field}.mention_id does not resolve")
        if idea_id not in idea_ids:
            raise LayeredOutcomeError(f"{field}.idea_id does not resolve")
        if relation not in LINK_RELATIONS:
            raise LayeredOutcomeError(f"{field}.relation is unsupported")
        identity = (mention_id, idea_id, relation)
        if identity in unique_links:
            raise LayeredOutcomeError(f"{field} duplicates an existing relation")
        unique_links.add(identity)
        linked_mentions.add(mention_id)
        linked_ideas.add(idea_id)
        if relation == "expresses":
            directly_expressed_ideas.add(idea_id)
        normalized_links.append({"mention_id": mention_id, "idea_id": idea_id, "relation": relation})
    if mention_ids - linked_mentions:
        raise LayeredOutcomeError(f"unlinked mentions: {sorted(mention_ids - linked_mentions)}")
    if idea_ids - linked_ideas:
        raise LayeredOutcomeError(f"ideas without transcript mentions: {sorted(idea_ids - linked_ideas)}")
    if schema_version == 2 and idea_ids - directly_expressed_ideas:
        raise LayeredOutcomeError(f"ideas without a direct expression anchor: {sorted(idea_ids - directly_expressed_ideas)}")

    result = dict(record)
    result["mentions"] = normalized_mentions
    result["ideas"] = normalized_ideas
    result["links"] = normalized_links
    result["validation"] = {
        "structural_valid": True,
        "direct_expression_anchor_validated": schema_version == 2,
        "exact_quote_integrity": True,
        "mention_count": len(normalized_mentions),
        "idea_count": len(normalized_ideas),
        "link_count": len(normalized_links),
        "evidence_quote_count": evidence_count,
        "semantic_entailment_validated": False,
        "inventory_completeness_validated": False,
    }
    return result


def idea_targets(record: dict[str, Any], *, forecastable_only: bool = False) -> list[dict[str, Any]]:
    """Project the abstract idea layer for outcome evaluation/training."""
    ideas = record.get("ideas", [])
    if forecastable_only:
        return [idea for idea in ideas if idea.get("forecastable") is True]
    return list(ideas)


def mentions_for_idea(record: dict[str, Any], idea_id: str) -> list[dict[str, Any]]:
    """Return transcript mentions linked to an idea, with their relation types."""
    mention_by_id = {mention["mention_id"]: mention for mention in record.get("mentions", [])}
    return [
        {**mention_by_id[link["mention_id"]], "relation": link["relation"]}
        for link in record.get("links", [])
        if link.get("idea_id") == idea_id
    ]
