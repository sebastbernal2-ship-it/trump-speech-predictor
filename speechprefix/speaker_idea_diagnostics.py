"""Failure decomposition and post-hoc prevalence calibration diagnostics."""

from __future__ import annotations
from collections import Counter
from datetime import date
from statistics import median
from typing import Any

from .speaker_idea_forecast import _rank_history


def analyze_issue_forecast_errors(test_predictions: list[dict[str, Any]], *, k: int = 10) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    false_positive_counts: Counter[str] = Counter()
    false_negative_counts: Counter[str] = Counter()
    unseen_counts: Counter[str] = Counter()
    lower_rank_counts: Counter[str] = Counter()
    displays: dict[str, str] = {}
    examples: list[dict[str, Any]] = []
    hit_raw: list[float] = []
    false_raw: list[float] = []
    hit_prevalence: list[float] = []
    false_prevalence: list[float] = []

    for event in test_predictions:
        targets = {item["idea_id"]: item for item in event["target_idea_diagnostics"]}
        predictions = event["top_10_predictions"][:k]
        predicted_ids = {item["idea_id"] for item in predictions}
        hits = predicted_ids & set(targets)
        false_positives = predicted_ids - set(targets)
        false_negatives = set(targets) - predicted_ids
        missed_unseen = {idea_id for idea_id in false_negatives if not targets[idea_id]["history_seen_before"]}
        missed_below = false_negatives - missed_unseen
        counts["events"] += 1
        counts["macro_precision_sum"] += len(hits) / k
        counts["macro_recall_sum"] += len(hits) / len(targets) if targets else 0.0
        counts["target_issue_phrase_instances"] += len(targets)
        counts["predicted_phrase_candidates"] += len(predictions)
        counts["exact_phrase_hits"] += len(hits)
        counts["false_positive_phrase_candidates"] += len(false_positives)
        counts["missed_target_phrase_instances"] += len(false_negatives)
        counts["missed_unseen_from_history"] += len(missed_unseen)
        counts["missed_seen_below_top_k"] += len(missed_below)
        counts["events_with_false_positives"] += bool(false_positives)
        counts["events_with_missed_targets"] += bool(false_negatives)
        counts["events_with_both_error_types"] += bool(false_positives and false_negatives)
        counts["events_with_no_hits"] += not bool(hits)
        counts["events_with_all_targets_covered"] += not bool(false_negatives)

        for prediction in predictions:
            idea_id = prediction["idea_id"]
            displays.setdefault(idea_id, prediction["idea"])
            if idea_id in targets:
                hit_raw.append(prediction["history_score"])
                hit_prevalence.append(prediction["weighted_history_prevalence"])
            else:
                false_raw.append(prediction["history_score"])
                false_prevalence.append(prediction["weighted_history_prevalence"])
                false_positive_counts[idea_id] += 1
        for idea_id in false_negatives:
            target = targets[idea_id]
            displays.setdefault(idea_id, target["idea"])
            false_negative_counts[idea_id] += 1
            if idea_id in missed_unseen:
                unseen_counts[idea_id] += 1
            else:
                lower_rank_counts[idea_id] += 1

        if false_positives or false_negatives:
            examples.append({
                "event_id": event["event_id"],
                "event_date": event["event_date"],
                "hit_count": len(hits),
                "false_positive_count": len(false_positives),
                "missed_target_count": len(false_negatives),
                "unseen_miss_count": len(missed_unseen),
                "target_ideas": [
                    {"idea": targets[idea_id]["idea"], "idea_id": idea_id,
                     "history_seen_before": targets[idea_id]["history_seen_before"],
                     "history_rank": targets[idea_id]["history_rank"],
                     "turn_ids": targets[idea_id]["supporting_turn_ids"]}
                    for idea_id in sorted(targets)
                ],
                "top_10_predictions": [
                    {"rank": item["rank"], "idea": item["idea"], "idea_id": item["idea_id"],
                     "history_score": item["history_score"],
                     "weighted_history_prevalence": item["weighted_history_prevalence"],
                     "matched_target": item["idea_id"] in targets}
                    for item in predictions
                ],
            })

    def top_phrases(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"idea": displays[idea_id], "idea_id": idea_id, "event_count": count}
            for idea_id, count in counter.most_common(limit)
        ]

    def score_summary(raw: list[float], prevalence: list[float]) -> dict[str, Any]:
        return {
            "count": len(raw),
            "mean_raw_history_score": sum(raw) / len(raw) if raw else 0.0,
            "median_raw_history_score": median(raw) if raw else 0.0,
            "mean_weighted_history_prevalence": sum(prevalence) / len(prevalence) if prevalence else 0.0,
        }

    examples.sort(key=lambda item: (item["hit_count"], -item["missed_target_count"], item["event_date"], item["event_id"]))
    event_count = counts["events"]
    return {
        "cutoff": k,
        "events": event_count,
        "target_issue_phrase_instances": counts["target_issue_phrase_instances"],
        "predicted_phrase_candidates": counts["predicted_phrase_candidates"],
        "exact_phrase_hits": counts["exact_phrase_hits"],
        "false_positive_phrase_candidates": counts["false_positive_phrase_candidates"],
        "missed_target_phrase_instances": counts["missed_target_phrase_instances"],
        "mean_targets_per_event": counts["target_issue_phrase_instances"] / event_count if event_count else 0.0,
        "mean_hits_per_event": counts["exact_phrase_hits"] / event_count if event_count else 0.0,
        "mean_false_positives_per_event": counts["false_positive_phrase_candidates"] / event_count if event_count else 0.0,
        "mean_missed_targets_per_event": counts["missed_target_phrase_instances"] / event_count if event_count else 0.0,
        "micro_precision_at_k": counts["exact_phrase_hits"] / (event_count * k) if event_count else 0.0,
        "micro_phrase_recall_at_k": counts["exact_phrase_hits"] / counts["target_issue_phrase_instances"] if counts["target_issue_phrase_instances"] else 0.0,
        "macro_precision_at_k": counts["macro_precision_sum"] / event_count if event_count else 0.0,
        "macro_event_recall_at_k": counts["macro_recall_sum"] / event_count if event_count else 0.0,
        "missed_target_decomposition": {
            "unseen_in_earlier_history": counts["missed_unseen_from_history"],
            "seen_in_history_but_ranked_below_cutoff": counts["missed_seen_below_top_k"],
        },
        "event_error_composition": {
            "with_false_positives": counts["events_with_false_positives"],
            "with_missed_targets": counts["events_with_missed_targets"],
            "with_both": counts["events_with_both_error_types"],
            "with_no_exact_hits": counts["events_with_no_hits"],
            "all_target_phrases_covered": counts["events_with_all_targets_covered"],
        },
        "top_false_positive_phrases": top_phrases(false_positive_counts),
        "top_missed_target_phrases": top_phrases(false_negative_counts),
        "top_unseen_missed_phrases": top_phrases(unseen_counts),
        "top_seen_but_low_rank_misses": top_phrases(lower_rank_counts),
        "top_10_raw_score_diagnostics": {
            "exact_phrase_hits": score_summary(hit_raw, hit_prevalence),
            "unmatched_predictions": score_summary(false_raw, false_prevalence),
            "raw_score_note": "history_score is a recency-weighted event count, not a confidence or probability",
        },
        "representative_failure_cases": examples[:8],
        "interpretation": [
            "A false positive means the normalized phrase was not mined from the target transcript; it does not establish that the idea was semantically absent.",
            "A missed target seen in history is a ranking/cutoff error for this model; an unseen target is not rankable by recurrence alone.",
            "Target phrases and phrase-level errors are automatic lexical diagnostics, not human-reviewed semantic judgments.",
        ],
    }


def evaluate_history_prevalence_calibration(
    events: list[dict[str, Any]], test_dates: set[date], *, half_life_days: int | None,
) -> dict[str, Any]:
    """Post-hoc calibration diagnostics for normalized event-frequency estimates."""
    weighted_pairs: list[tuple[float, int]] = []
    lifetime_pairs: list[tuple[float, int]] = []
    event_brier_weighted: list[float] = []
    event_brier_lifetime: list[float] = []
    target_instances = oov_instances = 0
    vocabulary_sizes = []
    bins = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1),
            (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.000001))
    bin_values: dict[tuple[float, float], list[tuple[float, int]]] = {item: [] for item in bins}

    for day in sorted(test_dates):
        history = [event for event in events if event["date"] < day]
        targets = [event for event in events if event["date"] == day]
        _, weighted_scores, _, total_weight = _rank_history(history, day, half_life_days=half_life_days)
        _, lifetime_scores, _, _ = _rank_history(history, day, half_life_days=None)
        vocabulary = sorted(weighted_scores)
        vocabulary_sizes.append(len(vocabulary))
        for target_event in targets:
            target = target_event["idea_ids"]
            target_instances += len(target)
            oov_instances += len(target - set(vocabulary))
            weighted_losses = []
            lifetime_losses = []
            for idea_id in vocabulary:
                observed = int(idea_id in target)
                weighted_probability = weighted_scores[idea_id] / total_weight if total_weight else 0.0
                lifetime_probability = lifetime_scores[idea_id] / len(history) if history else 0.0
                weighted_pairs.append((weighted_probability, observed))
                lifetime_pairs.append((lifetime_probability, observed))
                weighted_losses.append((weighted_probability - observed) ** 2)
                lifetime_losses.append((lifetime_probability - observed) ** 2)
                for interval in bins:
                    if interval[0] <= weighted_probability < interval[1]:
                        bin_values[interval].append((weighted_probability, observed))
                        break
            if weighted_losses:
                event_brier_weighted.append(sum(weighted_losses) / len(weighted_losses))
                event_brier_lifetime.append(sum(lifetime_losses) / len(lifetime_losses))

    def brier(pairs: list[tuple[float, int]]) -> float:
        return sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs) if pairs else 0.0

    return {
        "policy": "post-hoc plug-in prevalence: weighted prior-event count divided by total prior-event weight; diagnostic only, not a fitted or calibrated forecast confidence",
        "candidate_vocabulary_policy": "only phrase IDs observed in earlier events for each prediction date; unseen targets counted separately",
        "candidate_event_pairs": len(weighted_pairs),
        "mean_candidate_vocabulary_size": sum(vocabulary_sizes) / len(vocabulary_sizes) if vocabulary_sizes else 0.0,
        "target_phrase_instances": target_instances,
        "target_instances_unseen_in_history": oov_instances,
        "target_unseen_rate": oov_instances / target_instances if target_instances else 0.0,
        "brier_scores": {
            "recency_weighted_prevalence": {
                "micro": brier(weighted_pairs),
                "macro_event": sum(event_brier_weighted) / len(event_brier_weighted) if event_brier_weighted else 0.0,
            },
            "lifetime_prevalence": {
                "micro": brier(lifetime_pairs),
                "macro_event": sum(event_brier_lifetime) / len(event_brier_lifetime) if event_brier_lifetime else 0.0,
            },
        },
        "reliability_bins": [
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "candidate_event_pairs": len(bin_values[(lower, upper)]),
                "mean_estimated_prevalence": sum(p for p, _ in bin_values[(lower, upper)]) / len(bin_values[(lower, upper)]) if bin_values[(lower, upper)] else None,
                "empirical_mention_rate": sum(y for _, y in bin_values[(lower, upper)]) / len(bin_values[(lower, upper)]) if bin_values[(lower, upper)] else None,
            }
            for lower, upper in bins
        ],
        "limitations": [
            "These are test-set calibration diagnostics for a simple historical-frequency estimate; no calibration parameters were fitted.",
            "Phrase outcomes are automatic lexical matches and omit target phrases not previously seen in history.",
            "Per-event candidate vocabularies vary over time, so Brier scores are not a semantic-accuracy measure.",
        ],
    }
