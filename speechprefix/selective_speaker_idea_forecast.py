"""Selective, abstaining speaker-history issue forecasts.

The high-precision operating point predicts only phrases repeated in many
strictly prior events. It is a lexical weak-label system, not semantic gold.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .context_theme_ranker import read_jsonl
from .headline_idea_matching import (
    ISSUE_PHRASE_EXTRACTOR_VERSION,
    _target_turns,
    extract_transcript_ideas,
    issue_phrase_lexicon_sha256,
)


MODEL_VERSION = "selective-prior-event-support-v1"
DEFAULT_WINDOW_DAYS = 90
DEFAULT_MIN_HISTORY_EVENTS = 50
DEFAULT_MAX_PREDICTIONS = 1
TRAIN_END_DATE = date(2024, 10, 10)
VALIDATION_START_DATE = date(2024, 10, 11)
VALIDATION_END_DATE = date(2025, 12, 24)
RESERVED_TEST_START_DATE = date(2026, 1, 6)


def _load_events(
    input_path: str | Path, *, exclude_on_or_after: date | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(input_path)
    ids = [row.get("event_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("event IDs must be unique nonempty strings")
    events = []
    for row in rows:
        try:
            day = date.fromisoformat(row.get("event_date", ""))
        except (TypeError, ValueError):
            raise ValueError(f"invalid event_date for {row.get('event_id')}") from None
        if exclude_on_or_after is not None and day >= exclude_on_or_after:
            continue
        turns = _target_turns(row)
        ideas = extract_transcript_ideas(turns) if turns else []
        if not ideas:
            continue
        events.append({
            "event_id": row["event_id"],
            "date": day,
            "ideas": ideas,
            "idea_ids": {item["idea_id"] for item in ideas},
        })
    return sorted(events, key=lambda item: (item["date"], item["event_id"]))


def _rank_supported_ideas(
    history: list[dict[str, Any]], as_of: date, *, window_days: int,
    min_history_events: int, max_predictions: int,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    latest: dict[str, date] = {}
    surfaces: dict[str, Counter[str]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for event in history:
        age = (as_of - event["date"]).days
        if age <= 0 or age > window_days:
            continue
        for idea in event["ideas"]:
            idea_id = idea["idea_id"]
            counts[idea_id] += 1
            latest[idea_id] = max(latest.get(idea_id, event["date"]), event["date"])
            surfaces.setdefault(idea_id, Counter())[idea["candidate_idea"]] += idea["occurrence_count"]
            evidence.setdefault(idea_id, []).append({
                "event_id": event["event_id"],
                "event_date": event["date"].isoformat(),
                "supporting_turn_ids": idea["supporting_turn_ids"],
            })
    ranked = sorted(
        (idea_id for idea_id, count in counts.items() if count >= min_history_events),
        key=lambda idea_id: (-counts[idea_id], -latest[idea_id].toordinal(), idea_id),
    )[:max_predictions]
    results = []
    for rank, idea_id in enumerate(ranked, 1):
        sources = sorted(evidence[idea_id], key=lambda row: (row["event_date"], row["event_id"]))
        results.append({
            "rank": rank,
            "idea_id": idea_id,
            "idea": max(surfaces[idea_id].items(), key=lambda row: (row[1], row[0]))[0],
            "prior_event_support_count": counts[idea_id],
            "latest_supporting_event_date": latest[idea_id].isoformat(),
            "supporting_events": sources[-10:],
            "score_type": "distinct_prior_event_count_not_probability",
        })
    return results


def forecast_selective_speaker_ideas(
    input_path: str | Path,
    as_of: str,
    *,
    output_path: str | Path | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_history_events: int = DEFAULT_MIN_HISTORY_EVENTS,
    max_predictions: int = DEFAULT_MAX_PREDICTIONS,
) -> dict[str, Any]:
    """Forecast only recurrent phrases supported by a strict pre-date history."""
    try:
        cutoff = date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise ValueError("as_of must be an ISO date") from None
    if window_days < 1 or min_history_events < 1 or max_predictions < 1:
        raise ValueError("window_days, min_history_events, and max_predictions must be positive")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite forecast: {output_path}")
    events = _load_events(input_path, exclude_on_or_after=cutoff)
    history = [event for event in events if event["date"] < cutoff]
    predictions = _rank_supported_ideas(
        history, cutoff, window_days=window_days,
        min_history_events=min_history_events, max_predictions=max_predictions,
    )
    report = {
        "schema_version": 1,
        "forecast_type": "selective-speaker-history-issue-phrase-forecast",
        "model_version": MODEL_VERSION,
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "target_extractor": {
            "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
            "lexicon_sha256": issue_phrase_lexicon_sha256(),
        },
        "as_of_date": cutoff.isoformat(),
        "history_policy": "only events strictly earlier than as_of_date; same-day events excluded",
        "config": {
            "window_days": window_days,
            "minimum_distinct_prior_events": min_history_events,
            "maximum_predictions": max_predictions,
        },
        "history_event_count": sum(
            event["date"] < cutoff and (cutoff - event["date"]).days <= window_days
            for event in history
        ),
        "predictions": predictions,
        "abstained": not predictions,
        "limitations": [
            "Candidate ideas are automatic lexical phrases, not human-reviewed semantic ideas.",
            "Support counts are not confidence probabilities; abstention can improve precision only by sharply reducing coverage.",
            "This history-only forecast uses no event title, same-day transcript, news, or post-cutoff transcript content; same-day and later events are skipped before issue extraction.",
            "Historical transcript observation/availability timestamps are not independently audited, so strict real-world as-of availability is not established.",
        ],
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite forecast: {output}") from None
    return report


def _selective_metrics(
    events: list[dict[str, Any]], period_dates: set[date], *, window_days: int,
    min_history_events: int, max_predictions: int,
) -> dict[str, Any]:
    hits = predictions = target_labels = covered_events = 0
    sum_event_recall = sum_event_precision = 0.0
    events_scored = events_with_prediction = 0
    per_date: dict[str, dict[str, int]] = {}
    for target in events:
        day = target["date"]
        if day not in period_dates:
            continue
        history = [event for event in events if event["date"] < day]
        ranked = _rank_supported_ideas(
            history, day, window_days=window_days,
            min_history_events=min_history_events, max_predictions=max_predictions,
        )
        predicted_ids = {item["idea_id"] for item in ranked}
        hit_count = len(predicted_ids & target["idea_ids"])
        count = len(ranked)
        hits += hit_count
        predictions += count
        target_labels += len(target["idea_ids"])
        covered_events += bool(count)
        events_scored += 1
        sum_event_recall += hit_count / len(target["idea_ids"]) if target["idea_ids"] else 0.0
        if count:
            events_with_prediction += 1
            sum_event_precision += hit_count / count
        stats = per_date.setdefault(day.isoformat(), {"hits": 0, "predictions": 0})
        stats["hits"] += hit_count
        stats["predictions"] += count
    return {
        "events": events_scored,
        "events_with_prediction": events_with_prediction,
        "event_coverage": covered_events / events_scored if events_scored else 0.0,
        "predictions": predictions,
        "predictions_per_event": predictions / events_scored if events_scored else 0.0,
        "hits": hits,
        "pooled_precision": hits / predictions if predictions else 0.0,
        "pooled_recall": hits / target_labels if target_labels else 0.0,
        "mean_per_event_recall": sum_event_recall / events_scored if events_scored else 0.0,
        "mean_precision_when_not_abstaining": (
            sum_event_precision / events_with_prediction if events_with_prediction else 0.0
        ),
        "target_label_count": target_labels,
        "date_cluster_count": len(per_date),
        "per_date_counts": per_date,
    }


def _cluster_bootstrap_precision(
    metrics: dict[str, Any], *, seed: int = 1907, resamples: int = 2000,
) -> dict[str, Any]:
    clusters = metrics["per_date_counts"]
    dates = sorted(clusters)
    if not dates:
        return {"date_clusters": 0, "resamples": resamples, "interval_95pct": [0.0, 0.0]}
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        selected = [rng.choice(dates) for _ in dates]
        predictions = sum(clusters[day]["predictions"] for day in selected)
        hits = sum(clusters[day]["hits"] for day in selected)
        values.append(hits / predictions if predictions else 0.0)
    values.sort()
    return {
        "date_clusters": len(dates),
        "resamples": resamples,
        "seed": seed,
        "interval_95pct": [values[int((resamples - 1) * 0.025)], values[int((resamples - 1) * 0.975)]],
    }


def evaluate_selective_speaker_idea_forecast(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    target_precision: float = 0.8,
    minimum_inner_predictions: int = 50,
) -> dict[str, Any]:
    """Select a selective operating point inside training dates, report validation only.

    The final chronological test is intentionally neither scored nor described.
    """
    if not 0 < target_precision <= 1 or minimum_inner_predictions < 1:
        raise ValueError("target_precision must be in (0, 1] and minimum_inner_predictions positive")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")
    # Preserve the original speaker-idea test boundary even if extractor changes
    # alter which events receive labels. Never extract or score test-period labels.
    events = _load_events(input_path, exclude_on_or_after=date(2025, 12, 25))
    events = [event for event in events if event["date"] <= VALIDATION_END_DATE]
    train_dates = sorted({event["date"] for event in events if event["date"] <= TRAIN_END_DATE})
    validation_dates = {
        event["date"] for event in events
        if VALIDATION_START_DATE <= event["date"] <= VALIDATION_END_DATE
    }
    if len(train_dates) < 20 or not validation_dates:
        raise ValueError("fixed train/validation windows need at least 20 train dates and nonempty validation dates")
    inner_cut = max(1, min(len(train_dates) - 1, int(len(train_dates) * 0.7)))
    inner_validation_dates = set(train_dates[inner_cut:])
    windows = (30, 60, 90, 180, 365)
    support_thresholds = (5, 10, 15, 20, 25, 30, 40, 50)
    output_caps = (1, 3, 5, 10)
    candidates = []
    for window in windows:
        for minimum in support_thresholds:
            for cap in output_caps:
                metrics = _selective_metrics(
                    events, inner_validation_dates, window_days=window,
                    min_history_events=minimum, max_predictions=cap,
                )
                if metrics["predictions"] >= minimum_inner_predictions and metrics["pooled_precision"] >= target_precision:
                    candidates.append({
                        "window_days": window,
                        "minimum_distinct_prior_events": minimum,
                        "maximum_predictions": cap,
                        "inner_metrics": metrics,
                    })
    if not candidates:
        raise ValueError(
            "training-only operating-point selection found no config meeting target precision and minimum predictions"
        )
    selected = max(candidates, key=lambda item: (
        item["inner_metrics"]["event_coverage"],
        item["inner_metrics"]["mean_per_event_recall"],
        item["inner_metrics"]["pooled_precision"],
        -item["window_days"],
        -item["minimum_distinct_prior_events"],
        -item["maximum_predictions"],
    ))
    window = selected["window_days"]
    minimum = selected["minimum_distinct_prior_events"]
    cap = selected["maximum_predictions"]
    validation_metrics = _selective_metrics(
        events, validation_dates, window_days=window,
        min_history_events=minimum, max_predictions=cap,
    )
    # A fixed top-one, 30-day history ranker is a low-coverage-matched reference
    # only for describing the selective trade-off; it is not a tuned competitor.
    baseline = _selective_metrics(
        events, validation_dates, window_days=30,
        min_history_events=1, max_predictions=1,
    )
    report = {
        "schema_version": 1,
        "evaluation_type": "training-selected-selective-speaker-idea-forecast",
        "model_version": MODEL_VERSION,
        "target_policy": "automatic compact issue phrases; weak lexical labels, not semantic gold",
        "target_extractor": {
            "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
            "lexicon_sha256": issue_phrase_lexicon_sha256(),
        },
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "scored_event_count_through_locked_validation_end": len(events),
        "selection_policy": (
            "Select on the final 30% of training dates only: require pooled precision >= target and at least "
            "minimum_inner_predictions, then maximize event coverage; tie-break by mean recall, precision, "
            "shorter window, higher support, and smaller output cap."
        ),
        "config_selection": {
            "target_precision": target_precision,
            "minimum_inner_predictions": minimum_inner_predictions,
            "inner_train_date_count": inner_cut,
            "inner_validation_date_count": len(inner_validation_dates),
            "selected": {
                "window_days": window,
                "minimum_distinct_prior_events": minimum,
                "maximum_predictions": cap,
            },
            "inner_metrics": selected["inner_metrics"],
            "feasible_config_count": len(candidates),
        },
        "split": {
            "policy": "fixed original speaker-idea date boundaries; extractor changes cannot move the reserved test set",
            "train_date_range": [min(train_dates).isoformat(), max(train_dates).isoformat()],
            "train_date_count": len(train_dates),
            "train_event_count": sum(event["date"] <= TRAIN_END_DATE for event in events),
            "inner_validation_date_range": [min(inner_validation_dates).isoformat(), max(inner_validation_dates).isoformat()],
            "validation_date_range": [VALIDATION_START_DATE.isoformat(), VALIDATION_END_DATE.isoformat()],
            "validation_date_count": len(validation_dates),
            "validation_event_count": sum(event["date"] in validation_dates for event in events),
            "reserved_test_start_date_exclusive_for_this_report": RESERVED_TEST_START_DATE.isoformat(),
        },
        "validation_metrics": validation_metrics,
        "validation_pooled_precision_date_cluster_bootstrap": _cluster_bootstrap_precision(validation_metrics),
        "validation_reference_top1_30day": baseline,
        "limitations": [
            "Validation labels are automatic phrase matches, not human-reviewed semantic ideas.",
            "The compact-issue-phrases-v2 target rules were revised after inspecting earlier 2026 test errors; this report scores only the fixed pre-test validation dates, but the target definition is not pristine.",
            "The validation period has been used for development and is not an untouched test set.",
            "The original 2026-01-06-and-later test period is excluded before outcome extraction and is intentionally not evaluated by this report.",
            "High precision is obtained by abstaining on most events; report coverage and recall alongside precision.",
            "The label lexicon and event transcripts may omit semantically relevant issues; these metrics do not establish semantic forecast accuracy.",
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
            raise FileExistsError(f"refusing to overwrite evaluation report: {output}") from None
    return report
