"""Standalone, evidence-timed idea forecasting layer.

This package is intentionally separate from transcript continuation and
word-generation models. Its contracts are the stable boundary for evidence
retrieval/import, ranking, and in-speech update components.
"""

from .contracts import (
    is_source_eligible_at,
    seal_pre_speech_forecast,
    validate_event_manifest,
    validate_idea_input_snapshot,
    validate_idea_update,
    validate_pre_speech_forecast,
    validate_source_evidence,
)
from .ledger import append_source_items, append_source_records, freeze_idea_input_snapshot, read_source_ledger
from .news import retrieve_google_news_evidence
from .posts import load_trump_post_export, normalize_trump_post_export
from .ranker import make_idea_proposal_set, rank_pre_speech_ideas, seal_idea_proposal_set, validate_idea_proposal_set

__all__ = [
    "append_source_items",
    "append_source_records",
    "freeze_idea_input_snapshot",
    "is_source_eligible_at",
    "load_trump_post_export",
    "make_idea_proposal_set",
    "normalize_trump_post_export",
    "read_source_ledger",
    "rank_pre_speech_ideas",
    "seal_pre_speech_forecast",
    "seal_idea_proposal_set",
    "retrieve_google_news_evidence",
    "validate_event_manifest",
    "validate_idea_input_snapshot",
    "validate_idea_proposal_set",
    "validate_idea_update",
    "validate_pre_speech_forecast",
    "validate_source_evidence",
]
