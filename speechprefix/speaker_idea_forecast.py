"""Walk-forward prediction of recurring issue phrases from speaker history."""

from __future__ import annotations
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from .context_theme_ranker import read_jsonl
from .headline_idea_matching import _target_turns, extract_transcript_ideas


HALF_LIFE_GRID_DAYS: tuple[int | None, ...] = (30, 90, 180, 365, 730, None)
DEFAULT_TOP_K = (4, 10, 25)


def _rank_history(
    history: list[dict[str, Any]], as_of: date, *, half_life_days: int | None,
    window_days: int | None = None,
) -> tuple[list[str], dict[str, float], dict[str, Counter[str]]]:
    scores: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = {}
    for event in history:
        age_days = (as_of - event["date"]).days
        if window_days is not None and age_days > window_days:
            continue
        weight = 1.0 if half_life_days is None else math.exp(-math.log(2) * age_days / half_life_days)
        for idea in event["ideas"]:
            idea_id = idea["idea_id"]
            scores[idea_id] += weight
            surfaces.setdefault(idea_id, Counter())[idea["candidate_idea"]] += idea["occurrence_count"]
    ranked = sorted(scores, key=lambda idea_id: (-scores[idea_id], idea_id))
    return ranked, dict(scores), surfaces


def _event_metrics(prediction: list[str], target: set[str], top_k: tuple[int, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in top_k:
        hits = len(set(prediction[:k]) & target)
        result[f"precision_at_{k}"] = hits / k
        result[f"recall_at_{k}"] = hits / len(target) if target else 0.0
    relevance = [int(idea_id in target) for idea_id in prediction[:10]]
    dcg = sum(relevance[index] / math.log2(index + 2) for index in range(len(relevance)))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(target))))
    result["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return result


def _evaluate_period(
    events: list[dict[str, Any]], period_dates: set[date], *, half_life_days: int | None,
    top_k: tuple[int, ...], include_predictions: bool, window_days: int | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    per_event: list[dict[str, Any]] = []
    for day in sorted(period_dates):
        # All events on one date share the same history; same-day outcomes are not observable yet.
        history = [event for event in events if event["date"] < day]
        targets = [event for event in events if event["date"] == day]
        ranking, scores, history_surfaces = _rank_history(
            history, day, half_life_days=half_life_days, window_days=window_days,
        )
        for target_event in targets:
            metric = _event_metrics(ranking, target_event["idea_ids"], top_k)
            scored.append({key: metric[key] for key in metric})
            event_result = {
                "event_id": target_event["event_id"],
                "event_date": day.isoformat(),
                "metrics": metric,
            }
            if include_predictions:
                event_result.update({
                    "target_issue_phrase_count": len(target_event["idea_ids"]),
                    "target_idea_turn_ids": target_event["idea_turn_ids"],
                    "top_10_predictions": [
                        {"rank": rank, "idea_id": idea_id,
                         "idea": max(history_surfaces[idea_id].items(), key=lambda item: (item[1], item[0]))[0],
                         "history_score": round(scores[idea_id], 6),
                         "matched_target": idea_id in target_event["idea_ids"],
                         "supporting_turn_ids": target_event["idea_turn_ids"].get(idea_id, [])}
                        for rank, idea_id in enumerate(ranking[:10], 1)
                    ],
                })
            per_event.append(event_result)
    metrics: dict[str, float | int] = {"events": len(scored)}
    if not scored:
        return metrics, per_event
    for key in scored[0]:
        metrics[key] = sum(row[key] for row in scored) / len(scored)
    return metrics, per_event


def _paired_cluster_bootstrap(
    model_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], *,
    metrics: tuple[str, ...], seed: int = 1907, resamples: int = 1000,
) -> dict[str, Any]:
    baseline_by_id = {row["event_id"]: row for row in baseline_rows}
    if {row["event_id"] for row in model_rows} != set(baseline_by_id):
        raise ValueError("paired bootstrap models must score the same events")
    deltas_by_date: dict[str, dict[str, list[float]]] = {}
    for row in model_rows:
        baseline = baseline_by_id[row["event_id"]]
        date_metrics = deltas_by_date.setdefault(row["event_date"], {metric: [] for metric in metrics})
        for metric in metrics:
            date_metrics[metric].append(row["metrics"][metric] - baseline["metrics"][metric])
    dates = sorted(deltas_by_date)
    if not dates or resamples < 1:
        raise ValueError("paired bootstrap needs test dates and positive resamples")
    result: dict[str, Any] = {
        "events": len(model_rows),
        "date_clusters": len(dates),
        "resamples": resamples,
        "seed": seed,
    }
    for metric_index, metric in enumerate(metrics):
        point = [value for day in dates for value in deltas_by_date[day][metric]]
        rng = random.Random(seed + metric_index)
        distribution = []
        for _ in range(resamples):
            sampled_dates = [rng.choice(dates) for _ in dates]
            sample = [value for day in sampled_dates for value in deltas_by_date[day][metric]]
            distribution.append(sum(sample) / len(sample))
        distribution.sort()
        result[metric] = {
            "mean_delta": sum(point) / len(point),
            "cluster_bootstrap_95pct_ci": [
                distribution[int((resamples - 1) * 0.025)],
                distribution[int((resamples - 1) * 0.975)],
            ],
        }
    return result


def evaluate_speaker_idea_forecast(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    half_life_grid_days: tuple[int | None, ...] = HALF_LIFE_GRID_DAYS,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Select a recency half-life on validation and score walk-forward on a later holdout.

    Outcomes are automatic lexicon-filtered issue phrases from speech candidates.
    The model uses only issue phrases from strictly earlier event dates.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("train_fraction must be in (0,1) and validation_fraction in [0,1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction plus validation_fraction must be less than 1")
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k must contain positive values")
    if not half_life_grid_days or any(h is not None and h <= 0 for h in half_life_grid_days):
        raise ValueError("half-life values must be positive days or None for lifetime prevalence")
    if len(set(half_life_grid_days)) != len(half_life_grid_days):
        raise ValueError("half-life grid values must be unique")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")

    raw_records = read_jsonl(input_path)
    ids = [record.get("event_id") for record in raw_records]
    if any(not isinstance(event_id, str) or not event_id for event_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("event IDs must be unique nonempty strings")
    events = []
    skipped_without_ideas = 0
    for record in raw_records:
        try:
            event_date = date.fromisoformat(record.get("event_date", ""))
        except (TypeError, ValueError):
            raise ValueError(f"invalid event_date for {record['event_id']}") from None
        target_turns = _target_turns(record)
        if not target_turns:
            skipped_without_ideas += 1
            continue
        ideas = extract_transcript_ideas(target_turns)
        if not ideas:
            skipped_without_ideas += 1
            continue
        events.append({
            "event_id": record["event_id"],
            "date": event_date,
            "idea_ids": {idea["idea_id"] for idea in ideas},
            "ideas": ideas,
            "idea_turn_ids": {idea["idea_id"]: idea["supporting_turn_ids"] for idea in ideas},
        })
    events.sort(key=lambda event: (event["date"], event["event_id"]))
    dates = sorted({event["date"] for event in events})
    if len(dates) < 10:
        raise ValueError("need at least 10 distinct event dates with target issue phrases")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * train_fraction)))
    validation_end = min(
        len(dates) - 1,
        max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction))),
    )
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    if not train_dates or not validation_dates or not test_dates:
        raise ValueError("chronological split produced an empty partition")

    validation_results = {}
    for half_life in half_life_grid_days:
        metrics, _ = _evaluate_period(
            events, validation_dates, half_life_days=half_life, top_k=top_k, include_predictions=False,
        )
        validation_results[str(half_life) if half_life is not None else "lifetime"] = metrics
    selected_half_life = max(
        half_life_grid_days,
        key=lambda value: (
            validation_results[str(value) if value is not None else "lifetime"].get("ndcg_at_10", 0.0),
            value if value is not None else float("inf"),
        ),
    )
    selected_test_metrics, test_predictions = _evaluate_period(
        events, test_dates, half_life_days=selected_half_life, top_k=top_k, include_predictions=True,
    )
    lifetime_test_metrics, lifetime_test_rows = _evaluate_period(
        events, test_dates, half_life_days=None, top_k=top_k, include_predictions=False,
    )
    fixed_window_test_metrics = {}
    fixed_window_test_rows = {}
    fixed_window_validation_metrics = {}
    for window_days in (30, 90):
        validation_metrics, _ = _evaluate_period(
            events, validation_dates, half_life_days=None, window_days=window_days,
            top_k=top_k, include_predictions=False,
        )
        test_metrics, test_rows = _evaluate_period(
            events, test_dates, half_life_days=None, window_days=window_days,
            top_k=top_k, include_predictions=False,
        )
        fixed_window_validation_metrics[str(window_days)] = validation_metrics
        fixed_window_test_metrics[str(window_days)] = test_metrics
        fixed_window_test_rows[str(window_days)] = test_rows
    bootstrap_metrics = ("precision_at_4", "precision_at_10", "recall_at_10", "ndcg_at_10")
    test_metric_differences = {
        "recency_weighted_vs_lifetime": _paired_cluster_bootstrap(
            test_predictions, lifetime_test_rows, metrics=bootstrap_metrics,
        ),
        **{
            f"recency_weighted_vs_{window}_day_window": _paired_cluster_bootstrap(
                test_predictions, fixed_window_test_rows[str(window)], metrics=bootstrap_metrics,
                seed=1907 + window,
            )
            for window in (30, 90)
        },
    }
    test_start = min(test_dates)
    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_type": "walk-forward-speaker-history-issue-phrase-ranking",
        "model_family": "exponentially recency-weighted issue phrase prevalence",
        "model_version": "speaker-idea-recency-v1",
        "task": "rank issue phrases likely to recur in a Trump speech from strictly earlier speech-candidate transcripts",
        "target_policy": "automatic compact issue phrases mined from target-speaker turns; not semantic gold",
        "history_policy": "prior event dates only; all events on the target date are excluded",
        "selection_policy": "choose half-life with highest validation mean nDCG@10; ties prefer longer half-life",
        "config": {
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "half_life_grid_days": list(half_life_grid_days),
            "top_k": list(top_k),
        },
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "input_event_count": len(raw_records),
        "scored_event_count": len(events),
        "skipped_event_count_without_idea_labels": skipped_without_ideas,
        "idea_vocabulary_size": len({idea for event in events for idea in event["idea_ids"]}),
        "split": {
            "train_event_count": sum(event["date"] in train_dates for event in events),
            "validation_event_count": sum(event["date"] in validation_dates for event in events),
            "test_event_count": sum(event["date"] in test_dates for event in events),
            "train_date_range": [min(train_dates).isoformat(), max(train_dates).isoformat()],
            "validation_date_range": [min(validation_dates).isoformat(), max(validation_dates).isoformat()],
            "test_date_range": [test_start.isoformat(), max(test_dates).isoformat()],
            "train_event_ids": [event["event_id"] for event in events if event["date"] in train_dates],
            "validation_event_ids": [event["event_id"] for event in events if event["date"] in validation_dates],
            "test_event_ids": [event["event_id"] for event in events if event["date"] in test_dates],
        },
        "validation_metrics_by_half_life": validation_results,
        "validation_metrics_fixed_day_windows": fixed_window_validation_metrics,
        "selected_half_life_days": selected_half_life,
        "test_metrics": {
            "recency_weighted_model": selected_test_metrics,
            "lifetime_prevalence_baseline": lifetime_test_metrics,
            "fixed_day_window_baselines": fixed_window_test_metrics,
        },
        "test_metric_differences": test_metric_differences,
        "test_predictions": test_predictions,
        "limitations": [
            "Test outcomes are automatic phrase matches under a limited lexicon, not human-reviewed semantic ideas.",
            "This models recurring speaker-history issues; it does not use news context or predict exact wording.",
            "The test is a chronological walk-forward over the archive and can support this narrow ranking task only.",
            "Source publication/observation times for historical transcripts are not independently audited.",
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
