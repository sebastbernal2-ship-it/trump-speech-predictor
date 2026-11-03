"""Extract source-grounded context idea candidates from dated news headlines.

These are heuristic candidate-generation/evidence features, not semantic
probabilities. The language stays attached to the cited headline sources.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
import re
from typing import Any


TOPICS = {
    "election_turnout": {
        "summary": "Election stakes, turnout, voting, and campaign mobilization",
        "pattern": re.compile(r"\b(election|electoral|vote|voting|ballot|poll|polls|turnout|caucus|campaign)\b", re.I),
    },
    "working_class_economy": {
        "summary": "Working-class economic concerns, jobs, prices, wages, and trade",
        "pattern": re.compile(r"\b(working[- ]class|econom(?:y|ic)|jobs?|workers?|wages?|prices?|inflation|trade|manufacturing|tariffs?|tax(?:es)?)\b", re.I),
    },
    "immigration_border": {
        "summary": "Immigration, border enforcement, and migration policy",
        "pattern": re.compile(r"\b(immigration|immigrant|migrant|migration|border|deportation|asylum)\b", re.I),
    },
    "biden_garbage_controversy": {
        "summary": "Biden's 'garbage' remark and political response to it",
        "pattern": re.compile(r"\bgarbage\b", re.I),
        "requires_any": re.compile(r"\b(biden|trump|comment|remark|uproar|controversy)\b", re.I),
    },
    "media_press": {
        "summary": "News media, press coverage, and information disputes",
        "pattern": re.compile(r"\b(media|press|journalist|censorship|censor|broadcast|television|cbs|news outlet|news organization)\b", re.I),
    },
    "gaza_hostages_ceasefire": {
        "summary": "Gaza, hostages, ceasefire efforts, and regional mediation",
        "pattern": re.compile(r"\b(hostage|hostages|hamas|gaza|ceasefire|israel)\b", re.I),
    },
    "gulf_partnership_diplomacy": {
        "summary": "Gulf diplomacy, bilateral ties, and international partnership",
        "pattern": re.compile(r"\b(qatar|gulf|diplomacy|diplomatic|partnership|bilateral|alliance|relations)\b", re.I),
    },
    "business_investment": {
        "summary": "Commercial ties, investment, and business agreements",
        "pattern": re.compile(r"\b(investment|commercial|business|contract|dealmaking)\b", re.I),
    },
    "foreign_security": {
        "summary": "Foreign affairs, military strength, war, and national security",
        "pattern": re.compile(r"\b(war|military|iran|russia|ukraine|nato|foreign policy|national security|terrorism)\b", re.I),
    },
    "crime_public_safety": {
        "summary": "Crime, law enforcement, and public safety",
        "pattern": re.compile(r"\b(crime|criminal|police|public safety|violent|violence|gang|drug trafficking)\b", re.I),
    },
    "health_public_health": {
        "summary": "Health care and public-health issues",
        "pattern": re.compile(r"\b(health care|healthcare|medicare|medicaid|covid|vaccine|hospital|public health)\b", re.I),
    },
    "energy_environment": {
        "summary": "Energy production, environmental rules, and climate policy",
        "pattern": re.compile(r"\b(energy|climate|environment|electric vehicle|evs?|oil|gas drilling)\b", re.I),
    },
    "disaster_local_resources": {
        "summary": "Disaster response and local natural-resource issues",
        "pattern": re.compile(r"\b(hurricane|flood|wildfire|disaster|water resources?)\b", re.I),
    },
    "education_culture_values": {
        "summary": "Education, religion, cultural identity, and social values",
        "pattern": re.compile(r"\b(education|school|religion|christian|gender|family|abortion|culture|heritage|monument)\b", re.I),
    },
}


def extract_context_candidates(sources: list[dict[str, Any]], as_of: str) -> list[dict[str, Any]]:
    """Group dated pre-event headlines into heuristic topic candidates.

    A source may support more than one candidate. The returned evidence_score
    ranks source support and recency only; it is not a probability of speech coverage.
    """
    try:
        cutoff = date.fromisoformat(as_of)
    except ValueError:
        raise ValueError("as_of must be an ISO date (YYYY-MM-DD)") from None
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"sources": [], "seen": set()})
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        title = source.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("each source needs a nonempty title")
        raw_date = source.get("published_at") or source.get("published_date")
        try:
            published = date.fromisoformat(str(raw_date)[:10])
        except (TypeError, ValueError):
            raise ValueError("each source needs an ISO publication date") from None
        if published >= cutoff:
            raise ValueError(f"same-day or later source is ineligible: {title}")
        source_url = source.get("source_url") or source.get("url") or source.get("source_id")
        if not isinstance(source_url, str) or not source_url:
            raise ValueError("each source needs a URL or source ID")
        normalized = {
            "title": title.strip(),
            "publisher": str(source.get("publisher", "")).strip(),
            "published_at": published.isoformat(),
            "url": source_url,
        }
        for topic_id, topic in TOPICS.items():
            if not topic["pattern"].search(title):
                continue
            requires_any = topic.get("requires_any")
            if requires_any is not None and not requires_any.search(title):
                continue
            bucket = grouped[topic_id]
            if source_url not in bucket["seen"]:
                bucket["seen"].add(source_url)
                bucket["sources"].append(normalized)

    candidates = []
    lookback_days = max(1, (cutoff - min(
        date.fromisoformat(item["published_at"])
        for bucket in grouped.values() for item in bucket["sources"]
    )).days) if grouped else 1
    for topic_id, bucket in grouped.items():
        evidence = sorted(bucket["sources"], key=lambda item: (item["published_at"], item["title"]), reverse=True)
        publishers = {item["publisher"].casefold() for item in evidence if item["publisher"]}
        freshness = sum(
            max(0.0, 1.0 - (cutoff - date.fromisoformat(item["published_at"])).days / lookback_days)
            for item in evidence
        ) / len(evidence)
        candidates.append({
            "candidate_id": topic_id,
            "candidate_theme": TOPICS[topic_id]["summary"],
            "evidence_score": round(math.log1p(len(evidence)) + 0.25 * math.log1p(len(publishers)) + 0.25 * freshness, 6),
            "score_type": "untrained_source_evidence_heuristic_not_probability",
            "source_count": len(evidence),
            "publisher_count": len(publishers),
            "supporting_sources": evidence,
        })
    return sorted(candidates, key=lambda item: (-item["evidence_score"], item["candidate_id"]))
