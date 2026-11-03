"""Chronological evaluation for the weak context-to-theme-family ranker."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_theme_ranker import BernoulliThemeRanker, read_jsonl, tokenize_headlines


def _metrics(
    predictions: list[list[dict[str, Any]]],
    references: list[set[str]],
    categories: list[str],
    top_k: int = 4,
):
    if len(predictions) != len(references):
        raise ValueError("prediction and reference event counts differ")
    per_category = {}
    tp_total = fp_total = fn_total = 0
    brier_sum = 0.0
    for category in categories:
        tp = fp = fn = 0
        for ranked, reference in zip(predictions, references):
            probabilities = {item["theme_family"]: item["score"] for item in ranked}
            predicted = probabilities.get(category, 0.0) >= 0.5
            actual = category in reference
            tp += predicted and actual
            fp += predicted and not actual
            fn += not predicted and actual
            brier_sum += (probabilities.get(category, 0.0) - float(actual)) ** 2
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_category[category] = {"precision_at_0_5": precision, "recall_at_0_5": recall, "support": tp + fn}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1 = []
    for stats in per_category.values():
        p, r = stats["precision_at_0_5"], stats["recall_at_0_5"]
        macro_f1.append(2 * p * r / (p + r) if p + r else 0.0)
    topk_precision, topk_recall, exact = [], [], 0
    for ranked, reference in zip(predictions, references):
        top = {item["theme_family"] for item in ranked[:top_k]}
        topk_precision.append(len(top & reference) / top_k if top_k else 0.0)
        topk_recall.append(len(top & reference) / len(reference) if reference else 0.0)
        exact += top == reference
    n_events = len(references)
    return {
        "events_scored": n_events,
        "category_count": len(categories),
        "micro_precision_at_0_5": precision,
        "micro_recall_at_0_5": recall,
        "micro_f1_at_0_5": f1,
        "macro_f1_at_0_5": sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0,
        "brier_score_uncalibrated": brier_sum / (n_events * len(categories)) if n_events and categories else 0.0,
        f"mean_precision_at_{top_k}": sum(topk_precision) / n_events if n_events else 0.0,
        f"mean_recall_at_{top_k}": sum(topk_recall) / n_events if n_events else 0.0,
        "exact_top_k_set_match": exact / n_events if n_events else 0.0,
        "per_category": per_category,
    }


def evaluate_context_theme_ranker(
    snapshots_path: str | Path,
    labels_path: str | Path,
    *,
    output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Run a fixed chronological train/validation/test diagnostic and optionally save its report."""
    snapshots = read_jsonl(snapshots_path)
    labels = read_jsonl(labels_path)
    snapshot_by_id = {row.get("event_id"): row for row in snapshots}
    label_by_id = {row.get("event_id"): row for row in labels}
    if len(snapshot_by_id) != len(snapshots) or len(label_by_id) != len(labels):
        raise ValueError("duplicate event IDs in context or label input")
    if set(snapshot_by_id) != set(label_by_id):
        raise ValueError("context and theme-label files must cover the same event IDs")
    if not labels:
        raise ValueError("theme-family label file is empty")
    definitions = labels[0].get("family_definitions")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("theme-family definitions must be a nonempty object")
    definition_signature = json.dumps(definitions, sort_keys=True, ensure_ascii=False)
    if any(json.dumps(row.get("family_definitions"), sort_keys=True, ensure_ascii=False) != definition_signature for row in labels):
        raise ValueError("theme-family definitions differ across label records")
    categories = sorted(definitions)
    events = []
    for event_id, snapshot in snapshot_by_id.items():
        target = label_by_id[event_id]
        if snapshot.get("event_date") != target.get("event_date"):
            raise ValueError(f"event-date mismatch for {event_id}")
        if snapshot.get("exploratory_only") is not True:
            raise ValueError(f"context record is not explicitly marked exploratory: {event_id}")
        cutoff = date.fromisoformat(snapshot["prediction_cutoff_exclusive"])
        for source in snapshot.get("candidate_sources", []):
            published = date.fromisoformat(source["published_at"])
            if published >= cutoff:
                raise ValueError(f"non-pre-event context source for {event_id}")
        positives = target.get("positive_families")
        if not isinstance(positives, list):
            raise ValueError(f"invalid theme-family labels for {event_id}")
        if any(category not in definitions for category in positives):
            raise ValueError(f"unknown theme family in labels for {event_id}")
        if target.get("review_status") != "draft":
            raise ValueError(f"this pilot expects explicitly draft assistant labels for {event_id}")
        events.append({
            "event_id": event_id,
            "date": snapshot["event_date"],
            "document": tokenize_headlines(snapshot),
            "labels": set(positives),
            "snapshot": snapshot,
        })
    events.sort(key=lambda row: (row["date"], row["event_id"]))
    if len(events) < 10:
        raise ValueError("need at least 10 paired events for this chronological diagnostic")
    unique_dates = sorted({row["date"] for row in events})
    train_end = min(len(unique_dates) - 2, max(1, int(len(unique_dates) * train_fraction)))
    validation_end = min(
        len(unique_dates) - 1,
        max(train_end + 1, int(len(unique_dates) * (train_fraction + validation_fraction))),
    )
    train_dates = set(unique_dates[:train_end])
    validation_dates = set(unique_dates[train_end:validation_end])
    test_dates = set(unique_dates[validation_end:])
    if not train_dates or not validation_dates or not test_dates:
        raise ValueError("chronological split must produce nonempty train, validation, and test sets")
    train = [row for row in events if row["date"] in train_dates]
    validation = [row for row in events if row["date"] in validation_dates]
    test = [row for row in events if row["date"] in test_dates]
    ranker = BernoulliThemeRanker(alpha=1.0).fit(
        [row["document"] for row in train], [row["labels"] for row in train], categories,
    )

    def predict_rows(rows):
        return [ranker.predict(row["document"]) for row in rows]

    test_predictions = predict_rows(test)
    prevalence = {
        category: (ranker.category_counts[category] + 1) / (len(train) + 2)
        for category in categories
    }
    prior_predictions = [
        sorted(({"theme_family": category, "score": probability, "score_type": "training_prevalence"}
                for category, probability in prevalence.items()),
               key=lambda item: (-item["score"], item["theme_family"]))
        for _ in test
    ]
    result = {
        "schema_version": 1,
        "model_family": "context-headline-bernoulli-naive-bayes",
        "model_version": "diagnostic-v1",
        "task": "pre-speech broad-theme-family multi-label ranking",
        "label_policy": "assistant-authored weak family mappings; not human-reviewed semantic labels",
        "feature_policy": "dated source headlines only; event titles, transcript text, and theme summaries are excluded from model inputs",
        "context_policy": "all input sources are strictly earlier than event date; event location availability is unverified, so this dataset remains exploratory",
        "probability_policy": "Bernoulli NB posteriors are uncalibrated ranking scores; do not interpret as reliable probabilities",
        "config": {"alpha": ranker.alpha, "top_k": 4, "train_fraction": train_fraction, "validation_fraction": validation_fraction},
        "data_fingerprints": {
            "snapshots_sha256": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        },
        "split": {
            "train_events": [row["event_id"] for row in train],
            "validation_events": [row["event_id"] for row in validation],
            "test_events": [row["event_id"] for row in test],
            "train_dates": sorted(train_dates),
            "validation_dates": sorted(validation_dates),
            "test_dates": sorted(test_dates),
        },
        "train_label_support": ranker.category_counts,
        "test_metrics": {
            "context_naive_bayes": _metrics(test_predictions, [row["labels"] for row in test], categories),
            "training_prevalence_baseline": _metrics(prior_predictions, [row["labels"] for row in test], categories),
        },
        "validation_predictions_not_used_for_tuning": [
            {"event_id": row["event_id"], "date": row["date"], "top_families": predict_rows([row])[0][:4],
             "assistant_draft_targets": sorted(row["labels"])} for row in validation
        ],
        "test_predictions": [
            {"event_id": row["event_id"], "date": row["date"], "top_families": ranked[:4],
             "ranked_families": ranked, "assistant_draft_targets": sorted(row["labels"]),
             "supporting_sources": {
                 item["theme_family"]: [
                     {"title": source["title"], "publisher": source.get("publisher", ""),
                      "published_at": source["published_at"], "url": source["source_url"]}
                     for source in row["snapshot"]["candidate_sources"]
                     if any(token in tokenize_headlines({"candidate_sources": [source]}) for token in item["positive_feature_tokens"][:3])
                 ][:3]
                 for item in ranked[:4]
             }}
            for row, ranked in zip(test, test_predictions)
        ],
        "limitations": [
            "Only two chronological test events; metrics are highly unstable and are not semantic-accuracy claims.",
            "Theme families and transcript theme drafts are assistant-authored, unreviewed weak labels.",
            "The retrieved headlines are unreviewed candidates and can be noisy or irrelevant.",
            "Retrospective location hints have not been verified as available before the event.",
            "The fixed broad-family taxonomy cannot yet forecast arbitrary event-specific issue labels such as a particular controversy.",
        ],
    }
    digest_payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["report_content_sha256"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation artifact: {path}")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
