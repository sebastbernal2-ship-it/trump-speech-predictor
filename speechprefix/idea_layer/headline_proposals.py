"""Extractive, source-cited issue-phrase proposals from a frozen snapshot.

This is a lexical candidate generator, not a semantic idea model. It only emits
phrases in the existing issue lexicon and preserves the source IDs supporting
each phrase so it can feed the provider-neutral ranking pipeline.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

from .. import headline_idea_matching as _extractor
from ..headline_idea_matching import (
    ISSUE_PHRASE_EXTRACTOR_VERSION,
    extract_headline_ideas,
    issue_phrase_lexicon_sha256,
)
from .contracts import validate_idea_input_snapshot
from .ranker import make_idea_proposal_set

GENERATOR_VERSION = f"headline-issue-lexicon-{ISSUE_PHRASE_EXTRACTOR_VERSION}"


def _generator_provenance() -> str:
    module_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extractor_hash = hashlib.sha256(Path(_extractor.__file__).read_bytes()).hexdigest()
    return (
        f"{GENERATOR_VERSION};proposal_module_sha256={module_hash};"
        f"extractor_module_sha256={extractor_hash};"
        f"lexicon_sha256={issue_phrase_lexicon_sha256()}"
    )


def propose_headline_issue_phrases(
    snapshot: dict[str, Any],
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a sealed proposal set grounded in eligible news/post sources only.

    Event metadata sources are intentionally excluded from phrase extraction.
    Snapshot validation enforces source hashes, verified availability, and the
    exact cutoff; the phrase extractor then uses publication timestamps to group
    issue-anchor phrases and map them back to immutable snapshot source IDs.
    """
    validate_idea_input_snapshot(snapshot)
    cutoff = snapshot["prediction_cutoff_at"]
    proposal_sources: list[dict[str, Any]] = []
    for source in snapshot["sources"]:
        if source["source_type"] not in {"news", "trump_post", "other_primary_source"}:
            continue
        proposal_sources.append({
            "title": source["content"],
            "publisher": source["publisher"],
            "published_at": source["published_at"],
            "observed_at": source["observed_at"],
            # The legacy extractor carries this value through its evidence rows;
            # use the immutable ID, not a redirecting URL, as its identity key.
            "source_url": source["source_id"],
        })
    extracted = extract_headline_ideas(proposal_sources, cutoff, max_terms=2)
    proposals = []
    for idea in extracted:
        source_ids = sorted({row["url"] for row in idea["supporting_sources"]})
        proposals.append({
            "idea_id": idea["idea_id"],
            "idea": idea["candidate_idea"],
            "evidence_source_ids": source_ids,
        })
    return make_idea_proposal_set(
        snapshot,
        proposals,
        generator=_generator_provenance(),
        generated_at=generated_at,
    )
