"""Standalone, evidence-timed idea forecasting layer.

This package is intentionally separate from transcript continuation and
word-generation models. Its contracts are the stable boundary for future
retrieval, ranking, and in-speech update components.
"""

from .ledger import append_source_records, freeze_idea_input_snapshot, read_source_ledger
from .contracts import (
    is_source_eligible_at,
    validate_event_manifest,
    validate_idea_input_snapshot,
    validate_idea_update,
    validate_pre_speech_forecast,
    validate_source_evidence,
)

__all__ = [
    "append_source_records",
    "freeze_idea_input_snapshot",
    "is_source_eligible_at",
    "read_source_ledger",
    "validate_event_manifest",
    "validate_idea_input_snapshot",
    "validate_idea_update",
    "validate_pre_speech_forecast",
    "validate_source_evidence",
]
