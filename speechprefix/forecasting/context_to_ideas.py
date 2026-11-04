"""Supervised context-to-idea ranker trained on historical input/output pairs.

Features are strictly the target event's historical pre-event context headlines.
The target speech supplies labels only; it is never a prediction feature. Prior
speech labels are used to fit a supervised model on earlier examples, as in any
chronological training set, but prior speech text is not an inference input.
"""
from __future__ import annotations
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from ..context_theme_ranker import read_jsonl
from ..headline_idea_matching import ISSUE_PHRASE_EXTRACTOR_VERSION, _target_turns, extract_transcript_ideas, issue_phrase_lexicon_sha256

_TOKEN = re.compile(r"[a-z][a-z']{1,}")
_STOP = frozenset("""a an and are as at be been being by for from had has have he her his i in into is it its of on or our she that the their them there they this to was were will with we you your about after all also amid among around because before between both but can could during each even every first following further how however including just more most new not now only other over said says some such than then through today under until very while why would year years breaking latest live news report update watch video trump donald president campaign administration united states us""".split())


def _features(snapshot: dict[str, Any]) -> set[str]:
    sources = snapshot.get("candidate_sources")
    if not isinstance(sources, list):
        raise ValueError(f"{snapshot.get('event_id')}: candidate_sources must be a list")
    try:
        event_day = date.fromisoformat(snapshot.get("event_date", ""))
    except (TypeError, ValueError):
        raise ValueError(f"{snapshot.get('event_id')}: invalid event_date") from None
    lookback_start = event_day.fromordinal(event_day.toordinal() - 7)
    tokens: set[str] = set()
    for source in sources:
        title = source.get("title") if isinstance(source, dict) else None
        if not isinstance(title, str):
            raise ValueError(f"{snapshot.get('event_id')}: source is missing title")
        try:
            published_day = date.fromisoformat(str(source.get("published_at", ""))[:10])
        except (TypeError, ValueError):
            continue
        if not lookback_start <= published_day < event_day:
            continue
        words = [word for word in _TOKEN.findall(title.casefold()) if word not in _STOP]
        tokens.update("u:" + word for word in words)
        tokens.update("b:" + left + "_" + right for left, right in zip(words, words[1:]))
    return tokens


def _eligible_source_count(snapshot: dict[str, Any]) -> int:
    event_day = date.fromisoformat(snapshot["event_date"])
    start = date.fromordinal(event_day.toordinal() - 7)
    count = 0
    for source in snapshot.get("candidate_sources", []):
        try:
            published = date.fromisoformat(str(source.get("published_at", ""))[:10])
        except (TypeError, ValueError):
            continue
        count += start <= published < event_day
    return count


def _fit(documents: list[set[str]], labels: list[set[str]], *, alpha: float, min_df: int) -> dict[str, Any]:
    if not documents or len(documents) != len(labels):
        raise ValueError("training documents and labels must have equal nonzero length")
    if alpha <= 0 or min_df < 1:
        raise ValueError("alpha and min_df must be positive")
    df = Counter(feature for doc in documents for feature in doc)
    vocab = sorted(feature for feature, count in df.items() if count >= min_df)
    if not vocab:
        raise ValueError("headline vocabulary is empty")
    all_labels = sorted(set().union(*labels))
    models = {}
    for label in all_labels:
        positives = [i for i, target in enumerate(labels) if label in target]
        negatives = [i for i, target in enumerate(labels) if label not in target]
        n_pos, n_neg = len(positives), len(negatives)
        base = math.log((n_pos + alpha) / (n_neg + alpha))
        deltas = {}
        for feature in vocab:
            pos_count = sum(feature in documents[i] for i in positives)
            neg_count = sum(feature in documents[i] for i in negatives)
            p_pos = (pos_count + alpha) / (n_pos + 2 * alpha)
            p_neg = (neg_count + alpha) / (n_neg + 2 * alpha)
            absent = math.log1p(-p_pos) - math.log1p(-p_neg)
            present = math.log(p_pos) - math.log(p_neg)
            base += absent
            deltas[feature] = present - absent
        models[label] = {"base": base, "deltas": deltas, "support": n_pos}
    return {"vocabulary": vocab, "models": models, "alpha": alpha, "min_df": min_df}


def _predict(model: dict[str, Any], document: set[str]) -> list[tuple[str, float]]:
    features = document.intersection(model["vocabulary"])
    scores = []
    for label, params in model["models"].items():
        score = params["base"] + sum(params["deltas"][feature] for feature in features)
        scores.append((label, score))
    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))


def _metrics(ranked: list[tuple[str, float]], target: set[str], k: int = 10) -> dict[str, float]:
    predicted = [label for label, _ in ranked[:k]]
    hits = len(set(predicted) & target)
    relevance = [int(label in target) for label in predicted]
    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(target))))
    losses = []
    for label, logit in ranked:
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        probability = min(1.0 - 1e-12, max(1e-12, probability))
        outcome = int(label in target)
        losses.append(-(outcome * math.log(probability) + (1 - outcome) * math.log1p(-probability)))
    return {
        "precision_at_4": len(set(predicted[:4]) & target) / 4,
        "recall_at_10": hits / len(target) if target else 0.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "binary_log_loss": sum(losses) / len(losses) if losses else 0.0,
        "candidate_coverage": len(target & {label for label, _ in ranked}) / len(target) if target else 0.0,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {"events": 0}
    keys = ("precision_at_4", "recall_at_10", "ndcg_at_10", "binary_log_loss", "candidate_coverage")
    return {"events": len(results), **{key: sum(row["metrics"][key] for row in results) / len(results) for key in keys}}


def evaluate_context_to_idea_classifier(
    snapshots_path: str | Path,
    events_path: str | Path,
    *, output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Fit a headline-to-target-idea classifier on chronological examples."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("invalid chronological split fractions")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    snapshots = read_jsonl(snapshots_path)
    events = read_jsonl(events_path)
    snap_by_id = {row.get("event_id"): row for row in snapshots}
    event_by_id = {row.get("event_id"): row for row in events}
    if len(snap_by_id) != len(snapshots) or len(event_by_id) != len(events):
        raise ValueError("duplicate event IDs")
    if not snap_by_id or not set(snap_by_id) <= set(event_by_id):
        raise ValueError("every context snapshot needs a matching speech outcome")
    examples = []
    for event_id, snapshot in snap_by_id.items():
        event = event_by_id[event_id]
        day = snapshot.get("event_date")
        if day != event.get("event_date"):
            raise ValueError(f"event-date mismatch for {event_id}")
        cutoff = date.fromisoformat(day)
        if snapshot.get("exploratory_only") is not True:
            raise ValueError(f"snapshot {event_id} must be marked exploratory")
        for source in snapshot.get("candidate_sources", []):
            published = date.fromisoformat(str(source.get("published_at", ""))[:10])
            if published >= cutoff:
                raise ValueError(f"same-day/future source in {event_id}")
        ideas = extract_transcript_ideas(_target_turns(event))
        if not ideas:
            continue
        examples.append({
            "event_id": event_id, "date": cutoff, "profile": snapshot.get("retrieval_profile") or snapshot.get("retrieval_protocol"),
            "document": _features(snapshot), "targets": {idea["idea_id"] for idea in ideas},
            "target_ideas": ideas, "source_count": _eligible_source_count(snapshot),
        })
    examples.sort(key=lambda row: (row["date"], row["event_id"]))
    dates = sorted({row["date"] for row in examples})
    if len(dates) < 10:
        raise ValueError("need at least 10 distinct snapshot event dates")
    train_end = min(len(dates)-2, max(1, int(len(dates)*train_fraction)))
    val_end = min(len(dates)-1, max(train_end+1, int(len(dates)*(train_fraction+validation_fraction))))
    train_dates, val_dates, test_dates = set(dates[:train_end]), set(dates[train_end:val_end]), set(dates[val_end:])
    train = [row for row in examples if row["date"] in train_dates]
    validation = [row for row in examples if row["date"] in val_dates]
    test = [row for row in examples if row["date"] in test_dates]
    config_grid = [{"alpha": alpha, "min_df": min_df} for alpha in (0.25, 0.5, 1.0, 2.0, 4.0) for min_df in (1, 2, 3)]
    validation_rows = []
    for config in config_grid:
        model = _fit([row["document"] for row in train], [row["targets"] for row in train], **config)
        result = [_metrics(_predict(model, row["document"]), row["targets"]) for row in validation]
        validation_rows.append({"config": config, "metrics": _aggregate([{"metrics": item} for item in result])})
    selected = min(validation_rows, key=lambda row: (
        row["metrics"].get("binary_log_loss", float("inf")),
        -row["metrics"].get("ndcg_at_10", 0.0),
    ))
    model_config = selected["config"]
    # Refit on all pre-test examples; the test predictions remain chronological.
    # Incremental walk-forward refit: each target is scored only from examples dated earlier.
    test_rows = []
    for target in test:
        history = [row for row in train + validation + test if row["date"] < target["date"]]
        if not history:
            continue
        model_at_date = _fit([row["document"] for row in history], [row["targets"] for row in history], **model_config)
        ranking = _predict(model_at_date, target["document"])
        metrics = _metrics(ranking, target["targets"])
        test_rows.append({
            "event_id": target["event_id"], "event_date": target["date"].isoformat(),
            "retrieval_profile": target["profile"], "headline_count": target["source_count"],
            "target_idea_count": len(target["targets"]), "metrics": metrics,
            "target_ideas": sorted(target["targets"]),
            "top_10_predictions": [{"rank": rank, "idea_id": label, "score": round(score, 6),
                                   "matched_target": label in target["targets"]}
                                  for rank, (label, score) in enumerate(ranking[:10], 1)],
        })
    # Same chronological folds, no-context prior-prevalence reference (not a predictor feature).
    baseline_rows = []
    for target in test:
        history = [row for row in examples if row["date"] < target["date"]]
        counts = Counter(label for row in history for label in row["targets"])
        vocabulary = sorted(counts)
        ranked = []
        for label in vocabulary:
            positive = counts[label]
            negative = len(history) - positive
            ranked.append((label, math.log((positive + 1) / (negative + 1))))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        baseline_rows.append({"event_id": target["event_id"], "event_date": target["date"].isoformat(),
                              "metrics": _metrics(ranked, target["targets"])})
    final_model = _fit([row["document"] for row in examples], [row["targets"] for row in examples], **model_config)
    report: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "retrospective-context-to-idea-classifier-evaluation",
        "model_version": "bernoulli-headline-issue-classifier-v1",
        "task": "predict target-speech issue-phrase labels from dated news headlines published during the seven days before the target event; select the supervised model by binary cross-entropy loss then rank quality",
        "features": "headline title unigram/bigram presence from event_date minus 7 days through the day before event_date; no target transcript, prior speech text, event-title metadata, future context, or history count is an inference feature",
        "supervised_training": "earlier examples use their own preceding-week context headlines as features and their own transcript-derived ideas as labels; only examples earlier than each scored event are fit",
        "target_policy": "compact transcript-extracted lexical issue phrases; weak, incomplete labels rather than semantic gold",
        "context_availability": "snapshots were retrieved retrospectively; feed publication dates do not establish actual pre-event availability",
        "snapshot_sha256": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
        "event_corpus_sha256": hashlib.sha256(Path(events_path).read_bytes()).hexdigest(),
        "target_extractor": {"version": ISSUE_PHRASE_EXTRACTOR_VERSION, "lexicon_sha256": issue_phrase_lexicon_sha256()},
        "example_count": len(examples), "skipped_snapshot_count": len(snapshots)-len(examples),
        "split": {"train_event_ids":[row["event_id"] for row in train],"validation_event_ids":[row["event_id"] for row in validation],"test_event_ids":[row["event_id"] for row in test],
                  "train_dates":[min(train_dates).isoformat(),max(train_dates).isoformat()],"validation_dates":[min(val_dates).isoformat(),max(val_dates).isoformat()],"test_dates":[min(test_dates).isoformat(),max(test_dates).isoformat()]},
        "selection_policy": "select alpha and minimum headline document frequency by mean validation binary log loss over the training-label candidate vocabulary; break ties by nDCG@10; refit for each target using earlier examples only",
        "validation_grid": validation_rows, "selected_config": model_config,
        "test_metrics": {"headline_conditioned_classifier": _aggregate(test_rows), "training_label_prevalence_reference": _aggregate(baseline_rows)},
        "test_predictions": test_rows,
        "final_fit": {"fit_event_count":len(examples),"training_target_label_count":sum(len(row["targets"]) for row in examples),
                       "candidate_label_count":len(final_model["models"]),"vocabulary_size":len(final_model["vocabulary"]),
                       "model_sha256":hashlib.sha256(json.dumps(final_model,sort_keys=True,separators=(",",":")).encode()).hexdigest()},
        "limitations": ["Only 38 dated context/speech pairs are available in the merged snapshot set; chronological test results are tiny and unstable.",
                        "The historical Google News snapshots were retrieved after the target speeches, so timing/availability is unverified.",
                        "The prediction labels are a compact lexical extractor and omit many valid ideas or paraphrases.",
                        "Training labels from earlier speeches are supervision only; prior speech text or idea frequency is never supplied as a target-event predictor feature.",
                        "This is a retrospective development result; tuning on this small corpus makes reported selected scores optimistic."],
    }
    canonical=json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    report["report_content_sha256"]=hashlib.sha256(canonical.encode()).hexdigest()
    if output_path is not None:
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with path.open("x",encoding="utf-8") as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write("\n")
        except FileExistsError:raise FileExistsError(f"refusing to overwrite report: {output_path}") from None
    return report
