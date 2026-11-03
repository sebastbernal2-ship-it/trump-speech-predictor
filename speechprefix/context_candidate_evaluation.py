"""Chronological diagnostics for source-derived context idea candidates."""

from __future__ import annotations
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_candidates import TOPIC_FAMILY, extract_context_candidates
from .context_theme_ranker import read_jsonl


def _topk_metrics(predictions: list[list[str]], references: list[set[str]], k: int = 4) -> dict[str, float | int]:
    if len(predictions) != len(references):
        raise ValueError("prediction and reference event counts differ")
    if not references:
        return {"events_scored": 0, "precision_at_k": 0.0, "recall_at_k": 0.0}
    precision, recall = [], []
    for prediction, reference in zip(predictions, references):
        selected = set(prediction[:k])
        precision.append(len(selected & reference) / k)
        recall.append(len(selected & reference) / len(reference) if reference else 0.0)
    return {
        "events_scored": len(references),
        "precision_at_k": sum(precision) / len(precision),
        "recall_at_k": sum(recall) / len(recall),
    }


def evaluate_context_candidate_ranker(
    snapshots_path: str | Path,
    labels_path: str | Path,
    *,
    output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    snapshots = read_jsonl(snapshots_path)
    labels = read_jsonl(labels_path)
    snapshot_by_id = {row.get("event_id"): row for row in snapshots}
    label_by_id = {row.get("event_id"): row for row in labels}
    if len(snapshot_by_id) != len(snapshots) or len(label_by_id) != len(labels):
        raise ValueError("duplicate event IDs in input files")
    if not snapshots or set(snapshot_by_id) != set(label_by_id):
        raise ValueError("context and label files must contain the same nonempty event set")
    definitions = labels[0].get("family_definitions")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("labels need a nonempty theme-family taxonomy")
    signature = json.dumps(definitions, ensure_ascii=False, sort_keys=True)
    if any(json.dumps(row.get("family_definitions"), ensure_ascii=False, sort_keys=True) != signature for row in labels):
        raise ValueError("theme-family definitions differ across label rows")
    categories = sorted(definitions)

    events = []
    for event_id, snapshot in snapshot_by_id.items():
        label = label_by_id[event_id]
        if snapshot.get("event_date") != label.get("event_date"):
            raise ValueError(f"event-date mismatch for {event_id}")
        if snapshot.get("exploratory_only") is not True:
            raise ValueError(f"context must be marked exploratory for {event_id}")
        cutoff = date.fromisoformat(snapshot.get("prediction_cutoff_exclusive", snapshot["event_date"]))
        for source in snapshot.get("candidate_sources", []):
            published = date.fromisoformat(str(source.get("published_at", ""))[:10])
            if published >= cutoff:
                raise ValueError(f"same-day or later source for {event_id}")
        targets = label.get("positive_families")
        if label.get("review_status") != "draft" or not isinstance(targets, list):
            raise ValueError(f"expected assistant-draft labels for {event_id}")
        if not set(targets).issubset(categories):
            raise ValueError(f"unknown target family for {event_id}")
        candidates = extract_context_candidates(snapshot.get("candidate_sources", []), snapshot["event_date"])
        family_scores: dict[str, float] = {}
        family_evidence: dict[str, list[str]] = {}
        for candidate in candidates:
            for family in TOPIC_FAMILY[candidate["candidate_id"]]:
                family_scores[family] = max(family_scores.get(family, float("-inf")), candidate["evidence_score"])
                family_evidence.setdefault(family, []).append(candidate["candidate_id"])
        if not set(family_scores).issubset(categories):
            raise ValueError("candidate taxonomy references an unknown theme family")
        events.append({
            "event_id": event_id,
            "date": snapshot["event_date"],
            "targets": set(targets),
            "candidate_scores": family_scores,
            "candidate_evidence": family_evidence,
            "candidate_topics": candidates,
        })
    events.sort(key=lambda item: (item["date"], item["event_id"]))
    if len(events) < 10:
        raise ValueError("need at least 10 event-level pairs for chronological evaluation")
    dates = sorted({event["date"] for event in events})
    train_end = min(len(dates) - 2, max(1, int(len(dates) * train_fraction)))
    validation_end = min(len(dates) - 1, max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction))))
    train_dates, validation_dates, test_dates = set(dates[:train_end]), set(dates[train_end:validation_end]), set(dates[validation_end:])
    train = [event for event in events if event["date"] in train_dates]
    validation = [event for event in events if event["date"] in validation_dates]
    test = [event for event in events if event["date"] in test_dates]
    if not train or not validation or not test:
        raise ValueError("chronological split produced an empty partition")

    prevalence = {
        category: (sum(category in event["targets"] for event in train) + 1) / (len(train) + 2)
        for category in categories
    }
    evidence_predictions = [
        [family for family, _ in sorted(event["candidate_scores"].items(), key=lambda pair: (-pair[1], pair[0]))[:4]]
        for event in test
    ]
    prior_predictions = [
        [family for family, _ in sorted(prevalence.items(), key=lambda pair: (-pair[1], pair[0]))[:4]]
        for _ in test
    ]
    target_sets = [event["targets"] for event in test]
    all_candidate_sets = [set(event["candidate_scores"]) for event in test]
    unranked_coverage = [len(candidates & target) / len(target) if target else 0.0
                         for candidates, target in zip(all_candidate_sets, target_sets)]
    unranked_precision = [len(candidates & target) / len(candidates) if candidates else 0.0
                          for candidates, target in zip(all_candidate_sets, target_sets)]
    report = {
        "schema_version": 1,
        "model_family": "headline-candidate-evidence-ranker",
        "model_version": "diagnostic-v2",
        "task": "rank source-derived issue candidates for broad theme-family coverage",
        "candidate_policy": "fixed lexical issue patterns; a source can support multiple candidates",
        "score_policy": "evidence score uses source count, publisher diversity, and recency; it is not a learned or calibrated probability",
        "target_policy": "assistant-authored weak theme-family labels; not human-reviewed semantic labels",
        "context_policy": "all source publication dates are strictly before event dates; retrospective location hints remain unverified",
        "config": {"top_k": 4, "train_fraction": train_fraction, "validation_fraction": validation_fraction},
        "input_fingerprints": {
            "snapshots_sha256": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        },
        "split": {
            "train_event_ids": [event["event_id"] for event in train],
            "validation_event_ids": [event["event_id"] for event in validation],
            "test_event_ids": [event["event_id"] for event in test],
            "train_dates": sorted(train_dates), "validation_dates": sorted(validation_dates), "test_dates": sorted(test_dates),
        },
        "test_metrics": {
            "source_evidence_top4": _topk_metrics(evidence_predictions, target_sets),
            "training_prevalence_top4": _topk_metrics(prior_predictions, target_sets),
            "source_candidate_set_before_ranking": {
                "mean_precision": sum(unranked_precision) / len(unranked_precision),
                "mean_recall": sum(unranked_coverage) / len(unranked_coverage),
            },
        },
        "test_predictions": [
            {"event_id": event["event_id"], "date": event["date"],
             "top4_families": evidence_predictions[index],
             "all_candidates": event["candidate_topics"],
             "assistant_draft_targets": sorted(event["targets"])}
            for index, event in enumerate(test)
        ],
        "limitations": [
            "The holdout contains only two events and uses assistant-authored weak labels.",
            "The context retrieval pool is unreviewed and often noisy; event location availability is unverified.",
            "Lexical candidates are broad proxies and can miss ideas not represented in retrieved headlines.",
            "Evidence scores are not probabilities; the current experiment tests candidate coverage/ranking only.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation artifact: {output}")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
