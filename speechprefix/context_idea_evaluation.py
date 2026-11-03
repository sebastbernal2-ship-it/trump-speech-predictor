"""Chronological evaluation of context-derived idea candidates vs speaker history."""

from __future__ import annotations
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_theme_ranker import read_jsonl
from .headline_idea_matching import _target_turns, extract_headline_ideas, extract_transcript_ideas


def _metrics(predictions: list[list[str]], targets: list[set[str]], k: int) -> dict[str, float | int]:
    precisions, recalls = [], []
    for prediction, target in zip(predictions, targets):
        hits = len(set(prediction[:k]) & target)
        precisions.append(hits / k)
        recalls.append(hits / len(target) if target else 0.0)
    return {
        "events": len(targets),
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
    }


def evaluate_context_idea_candidates(
    snapshots_path: str | Path,
    event_records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    top_k: tuple[int, ...] = (4, 10, 25),
) -> dict[str, Any]:
    """Compare context phrase ranking with past-transcript phrase prevalence.

    Transcript phrase lists are automatic weak outcomes. Context sources must
    predate each event, but historical retrieval snapshots remain exploratory.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("train_fraction must be in (0,1) and validation_fraction in [0,1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction plus validation_fraction must be less than 1")
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k must contain positive values")

    snapshots = read_jsonl(snapshots_path)
    records = read_jsonl(event_records_path)
    snapshots_by_id = {row.get("event_id"): row for row in snapshots}
    records_by_id = {row.get("event_id"): row for row in records}
    if len(snapshots_by_id) != len(snapshots) or len(records_by_id) != len(records):
        raise ValueError("duplicate event IDs in input files")
    if not snapshots or not set(snapshots_by_id).issubset(records_by_id):
        raise ValueError("every context snapshot must have a matching transcript record")

    events = []
    for event_id, snapshot in snapshots_by_id.items():
        record = records_by_id[event_id]
        event_date = snapshot.get("event_date")
        if event_date != record.get("event_date"):
            raise ValueError(f"event-date mismatch for {event_id}")
        try:
            cutoff = date.fromisoformat(event_date)
        except (TypeError, ValueError):
            raise ValueError(f"invalid event_date for {event_id}") from None
        if snapshot.get("exploratory_only") is not True:
            raise ValueError(f"context snapshot must be marked exploratory for {event_id}")
        sources = snapshot.get("candidate_sources")
        if not isinstance(sources, list):
            raise ValueError(f"candidate_sources must be a list for {event_id}")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("title"), str):
                raise ValueError(f"invalid context source for {event_id}")
            try:
                published = date.fromisoformat(str(source.get("published_at", ""))[:10])
            except ValueError:
                raise ValueError(f"invalid source publication date for {event_id}") from None
            if published >= cutoff:
                raise ValueError(f"same-day or later context source for {event_id}")

        turns = _target_turns(record)
        if not turns:
            raise ValueError(f"no target-speaker turns for {event_id}")
        target_ideas = extract_transcript_ideas(turns)
        target_ids = {idea["idea_id"] for idea in target_ideas}
        context_ideas = [
            idea for idea in extract_headline_ideas(sources, event_date)
            if idea["term_count"] >= 2
        ]
        context_ideas.sort(key=lambda item: (-item["evidence_score"], -item["term_count"], item["idea_id"]))
        events.append({
            "event_id": event_id,
            "date": event_date,
            "retrieval_profile": snapshot.get("retrieval_profile") or snapshot.get("context_retrieval_protocol"),
            "targets": target_ids,
            "target_ideas": target_ideas,
            "context_ideas": context_ideas,
        })
    events.sort(key=lambda row: (row["date"], row["event_id"]))
    dates = sorted({row["date"] for row in events})
    if len(dates) < 10:
        raise ValueError("need at least 10 distinct event dates for chronological evaluation")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * train_fraction)))
    validation_end = min(
        len(dates) - 1,
        max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction))),
    )
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    train = [row for row in events if row["date"] in train_dates]
    validation = [row for row in events if row["date"] in validation_dates]
    test = [row for row in events if row["date"] in test_dates]
    if not train or not validation or not test:
        raise ValueError("chronological split produced an empty partition")

    phrase_event_counts: dict[str, int] = {}
    phrase_surfaces: dict[str, dict[str, int]] = {}
    for row in train:
        for idea in row["target_ideas"]:
            idea_id = idea["idea_id"]
            phrase_event_counts[idea_id] = phrase_event_counts.get(idea_id, 0) + 1
            surfaces = phrase_surfaces.setdefault(idea_id, {})
            surface = idea["candidate_idea"]
            surfaces[surface] = surfaces.get(surface, 0) + idea["occurrence_count"]
    prevalence_ranked = [
        phrase for phrase, _ in sorted(phrase_event_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    context_predictions = [
        [idea["idea_id"] for idea in row["context_ideas"]]
        for row in test
    ]
    prevalence_predictions = [prevalence_ranked for _ in test]
    target_sets = [row["targets"] for row in test]

    metrics = {}
    for k in top_k:
        metrics[f"top_{k}"] = {
            "context_evidence_ranker": _metrics(context_predictions, target_sets, k),
            "training_transcript_prevalence": _metrics(prevalence_predictions, target_sets, k),
        }
    test_predictions = []
    for index, row in enumerate(test):
        test_predictions.append({
            "event_id": row["event_id"],
            "date": row["date"],
            "retrieval_profile": row["retrieval_profile"],
            "context_top_10": [
                {"idea": idea["candidate_idea"], "idea_id": idea["idea_id"],
                 "source_count": idea["source_count"], "publisher_count": idea["publisher_count"],
                 "matched_target": idea["idea_id"] in row["targets"],
                 "supporting_sources": idea["supporting_sources"][:3]}
                for idea in row["context_ideas"][:10]
            ],
            "history_prevalence_top_10": [
                {"idea": max(phrase_surfaces[idea_id].items(), key=lambda item: (item[1], item[0]))[0],
                 "idea_id": idea_id, "training_event_count": phrase_event_counts[idea_id],
                 "matched_target": idea_id in row["targets"]}
                for idea_id in prevalence_ranked[:10]
            ],
            "target_issue_phrase_count": len(row["targets"]),
            "target_idea_turn_ids": {
                idea["idea_id"]: idea["supporting_turn_ids"]
                for idea in row["target_ideas"]
            },
        })
    report = {
        "schema_version": 1,
        "evaluation_type": "chronological-context-idea-candidates-vs-speaker-history",
        "model_family": "source-supported compact issue-phrase ranker",
        "model_version": "topic-family-retrieval-v1",
        "task": "rank context headline issue phrases and compare with past-transcript phrase prevalence",
        "context_ranking": "source count, publisher diversity, and recency heuristic; not trained or calibrated",
        "history_baseline": "distinct training events containing each automatically extracted issue phrase",
        "target_policy": "automatically mined, lexicon-filtered transcript issue phrases with cited turns; not human semantic gold",
        "context_policy": "all source dates are strictly before the event date; snapshots were retrieved retrospectively",
        "config": {"train_fraction": train_fraction, "validation_fraction": validation_fraction, "top_k": list(top_k)},
        "input_sha256": {
            "snapshots": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
            "event_records": hashlib.sha256(Path(event_records_path).read_bytes()).hexdigest(),
        },
        "split": {
            "train_event_ids": [row["event_id"] for row in train],
            "validation_event_ids": [row["event_id"] for row in validation],
            "test_event_ids": [row["event_id"] for row in test],
            "train_dates": sorted(train_dates),
            "validation_dates": sorted(validation_dates),
            "test_dates": sorted(test_dates),
        },
        "test_metrics": metrics,
        "test_predictions": test_predictions,
        "limitations": [
            "The test set contains only two events and supports no reliable generalization claim.",
            "Retrieved-after-event Google News results are not prospective input snapshots, even when article dates precede the event.",
            "Automatic phrase matches are lexical diagnostics, not semantic equivalence or human-reviewed labels.",
            "The context ranker is a source-support heuristic; the history baseline may benefit from recurring speaker topics.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation report: {output}")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
