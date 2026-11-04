"""Replay an untuned speaker-history baseline before assistant-reviewed events.

This module accepts only a corpus of dated transcripts and an event/date manifest.
It never reads transcript outcome annotations or review labels.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from ..context_theme_ranker import read_jsonl
from ..headline_idea_matching import (
    ISSUE_PHRASE_EXTRACTOR_VERSION,
    _target_turns,
    extract_transcript_ideas,
    issue_phrase_lexicon_sha256,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_history_prevalence_for_review_events(
    history_path: str | Path,
    targets_path: str | Path,
    *,
    output_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Generate fixed lifetime-prevalence rankings using only earlier event dates.

    The date manifest must contain only ``event_id`` and ``event_date``. All
    target-day and future corpus rows are discarded before transcript extraction.
    The rank is an uncalibrated count of distinct earlier events, not a probability.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite forecast artifact: {output_path}")

    targets_raw = read_jsonl(targets_path)
    targets: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for row in targets_raw:
        if set(row) != {"event_id", "event_date"}:
            raise ValueError("target manifest rows must contain only event_id and event_date")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen_targets:
            raise ValueError("target event IDs must be unique nonempty strings")
        try:
            cutoff = date.fromisoformat(row.get("event_date", ""))
        except (TypeError, ValueError):
            raise ValueError(f"invalid target event_date for {event_id}") from None
        seen_targets.add(event_id)
        targets.append({"event_id": event_id, "event_date": cutoff})

    records = read_jsonl(history_path)
    ids = [row.get("event_id") for row in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("history corpus event IDs must be unique nonempty strings")
    dated: list[tuple[date, dict[str, Any]]] = []
    date_by_id: dict[str, date] = {}
    for row in records:
        try:
            day = date.fromisoformat(row.get("event_date", ""))
        except (TypeError, ValueError):
            raise ValueError(f"invalid history event_date for {row.get('event_id')}") from None
        dated.append((day, row))
        date_by_id[row["event_id"]] = day

    forecasts: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda item: (item["event_date"], item["event_id"])):
        cutoff = target["event_date"]
        if target["event_id"] in date_by_id and date_by_id[target["event_id"]] != cutoff:
            raise ValueError(f"target date disagrees with history corpus for {target['event_id']}")
        # Critical leakage boundary: the target event, later events, and same-day
        # records are removed before target-turn parsing or issue extraction.
        eligible = [
            (day, row) for day, row in dated
            if row["event_id"] != target["event_id"] and day < cutoff
        ]
        history: list[dict[str, Any]] = []
        for day, row in eligible:
            turns = _target_turns(row)
            ideas = extract_transcript_ideas(turns) if turns else []
            if ideas:
                history.append({"event_id": row["event_id"], "date": day, "ideas": ideas})

        from ..speaker_idea_forecast import _rank_history

        ranking, scores, surfaces, _total_weight = _rank_history(
            history, cutoff, half_life_days=None,
        )
        support_by_id: dict[str, list[dict[str, Any]]] = {}
        for event in history:
            for idea in event["ideas"]:
                support_by_id.setdefault(idea["idea_id"], []).append({
                    "event_id": event["event_id"],
                    "event_date": event["date"].isoformat(),
                    "supporting_turn_ids": idea["supporting_turn_ids"],
                })
        candidates = []
        for rank, idea_id in enumerate(ranking[:top_k], 1):
            candidate_events = support_by_id[idea_id]
            candidates.append({
                "rank": rank,
                "idea_id": idea_id,
                "idea": max(surfaces[idea_id].items(), key=lambda item: (item[1], item[0]))[0],
                "prior_distinct_event_count": len(candidate_events),
                "score_type": "uncalibrated_lifetime_event_prevalence",
                "supporting_events_sample": candidate_events[-10:],
            })
        forecasts.append({
            "event_id": target["event_id"],
            "event_date": cutoff.isoformat(),
            "cutoff_policy": "exclusive event date; same-day transcripts excluded",
            "eligible_corpus_record_count": len(eligible),
            "history_events_with_extractable_ideas": len(history),
            "latest_eligible_history_date": max((day for day, _ in eligible), default=None).isoformat()
            if eligible else None,
            "predictions": candidates,
            "abstained": not candidates,
        })

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "assistant-reference-review-history-only-forecast-replay",
        "model_family": "fixed lifetime prevalence of automatic transcript issue phrases",
        "model_version": "speaker-idea-recency-v1-lifetime-baseline",
        "task": "rank recurring issue phrases from strictly earlier speech-candidate transcripts",
        "evaluation_mode": "retrospective chronological replay; not a prospective forecast",
        "history_input_sha256": _sha256(history_path),
        "target_manifest_sha256": _sha256(targets_path),
        "target_manifest_policy": "event_id and event_date only; no outcome labels accepted",
        "target_extractor": {
            "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
            "lexicon_sha256": issue_phrase_lexicon_sha256(),
        },
        "ranking_policy": "fixed lifetime distinct-event prevalence; descending count, then idea_id; no tuning",
        "top_k": top_k,
        "forecasts": forecasts,
        "limitations": [
            "Candidate phrases are automatic lexical issue labels, not semantic ideas; matches to assistant-reviewed references must be adjudicated separately.",
            "Scores are uncalibrated rankings, not probabilities.",
            "Historical transcript publication/availability timestamps are not independently audited; prior event date does not prove the transcript was observable by the cutoff.",
            "This retrospective replay is not a saved pre-speech forecast and must not be represented as prospective accuracy.",
            "Assistant-reviewed references are post-hoc interpretive references, not objective semantic gold; this report does not contain outcome labels.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite forecast artifact: {output}") from None
    return report
