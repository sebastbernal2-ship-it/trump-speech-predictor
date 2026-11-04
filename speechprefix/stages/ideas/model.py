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

from ...context_theme_ranker import read_jsonl
from ...headline_idea_matching import ISSUE_PHRASE_EXTRACTOR_VERSION, _target_turns, extract_transcript_ideas, issue_phrase_lexicon_sha256

_TOKEN = re.compile(r"[a-z][a-z']{1,}")
_STOP = frozenset("""a an and are as at be been being by for from had has have he her his i in into is it its of on or our she that the their them there they this to was were will with we you your about after all also amid among around because before between both but can could during each even every first following further how however including just more most new not now only other over said says some such than then through today under until very while why would year years breaking latest live news report update watch video trump donald president campaign administration united states us""".split())


def _features(
    snapshot: dict[str, Any], *, headline_limit: int | None = None, selection: str = "latest"
) -> set[str]:
    sources = snapshot.get("candidate_sources")
    if not isinstance(sources, list):
        raise ValueError(f"{snapshot.get('event_id')}: candidate_sources must be a list")
    try:
        event_day = date.fromisoformat(snapshot.get("event_date", ""))
    except (TypeError, ValueError):
        raise ValueError(f"{snapshot.get('event_id')}: invalid event_date") from None
    lookback_start = event_day.fromordinal(event_day.toordinal() - 7)
    eligible: list[tuple[date, int, str]] = []
    seen_titles: set[str] = set()
    for source_index, source in enumerate(sources):
        title = source.get("title") if isinstance(source, dict) else None
        if not isinstance(title, str):
            raise ValueError(f"{snapshot.get('event_id')}: source is missing title")
        try:
            published_day = date.fromisoformat(str(source.get("published_at", ""))[:10])
        except (TypeError, ValueError):
            continue
        if not lookback_start <= published_day < event_day:
            continue
        publisher = source.get("publisher")
        if isinstance(publisher, str) and publisher.strip():
            for separator in (" - ", " — ", " | "):
                suffix = separator + publisher.strip()
                if title.casefold().endswith(suffix.casefold()):
                    title = title[:-len(suffix)].strip()
                    break
        normalized_title = " ".join(title.casefold().split())
        if not normalized_title or normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        eligible.append((published_day, source_index, title))
    # Newest publication day first; preserve captured source order for date ties,
    # which often encode the retrieval/query ranking when feed times are coarse.
    eligible.sort(key=lambda row: (-row[0].toordinal(), row[1]))
    if headline_limit is not None:
        if selection == "latest":
            eligible = eligible[:headline_limit]
        elif selection == "spread_week":
            by_day: dict[date, list[tuple[date, int, str]]] = {}
            for item in eligible:
                by_day.setdefault(item[0], []).append(item)
            days = sorted(by_day, reverse=True)
            selected = []
            for rank in range(max((len(by_day[day]) for day in days), default=0)):
                for day in days:
                    if rank < len(by_day[day]) and len(selected) < headline_limit:
                        selected.append(by_day[day][rank])
            eligible = selected
        else:
            raise ValueError(f"unsupported headline selection strategy: {selection}")
    tokens: set[str] = set()
    for _, _, title in eligible:
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


def _metrics(
    ranked: list[tuple[str, float]],
    target: set[str],
    *,
    all_labels: set[str],
    k: int = 10,
) -> dict[str, float]:
    predicted = [label for label, _ in ranked[:k]]
    hits = len(set(predicted) & target)
    relevance = [int(label in target) for label in predicted]
    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(target))))
    # Score every idea label known before this target, including negatives. A
    # target idea unseen in training cannot be predicted and is counted as a miss,
    # not silently removed from the loss denominator.
    ranked_score = dict(ranked)
    losses = []
    for label in sorted(all_labels | target):
        logit = ranked_score.get(label, -40.0)
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        probability = min(1.0 - 1e-12, max(1e-12, probability))
        outcome = int(label in target)
        losses.append(-(outcome * math.log(probability) + (1 - outcome) * math.log1p(-probability)))
    return {
        "precision_at_4": len(set(predicted[:4]) & target) / 4,
        "recall_at_10": hits / len(target) if target else 0.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "binary_log_loss": sum(losses) / len(losses) if losses else 0.0,
        "candidate_coverage": len(target & all_labels) / len(target) if target else 0.0,
        "unseen_target_ideas": float(len(target - all_labels)),
    }


def _label_names(examples: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, Counter[str]] = {}
    for example in examples:
        for idea in example.get("target_ideas", []):
            idea_id = idea.get("idea_id")
            name = idea.get("candidate_idea")
            if isinstance(idea_id, str) and isinstance(name, str) and name.strip():
                names.setdefault(idea_id, Counter())[name.strip()] += int(idea.get("occurrence_count", 1))
    return {idea_id: counts.most_common(1)[0][0] for idea_id, counts in names.items()}


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {"events": 0}
    keys = ("precision_at_4", "recall_at_10", "ndcg_at_10", "binary_log_loss", "candidate_coverage", "unseen_target_ideas")
    return {"events": len(results), **{key: sum(row["metrics"][key] for row in results) / len(results) for key in keys}}


def predict_context_ideas(snapshot: dict[str, Any], evaluation_artifact: dict[str, Any], *, top_k: int = 10) -> list[dict[str, Any]]:
    """Predict ranked ideas for a later event from a fitted evaluation artifact.

    The artifact's final fit uses all available paired historical examples. This
    function accepts only the target event's context snapshot, never its transcript.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if evaluation_artifact.get("artifact_type") != "retrospective-context-to-idea-classifier-evaluation":
        raise ValueError("unsupported fitted-model artifact")
    final_fit = evaluation_artifact.get("final_model")
    if not isinstance(final_fit, dict) or not isinstance(final_fit.get("model"), dict):
        raise ValueError("evaluation artifact does not contain a fitted model")
    expected_model_hash = evaluation_artifact.get("final_fit", {}).get("model_sha256")
    actual_model_hash = hashlib.sha256(
        json.dumps(final_fit["model"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_model_hash != expected_model_hash:
        raise ValueError("fitted model hash mismatch")
    expected_report_hash = evaluation_artifact.get("report_content_sha256")
    report_without_hash = dict(evaluation_artifact)
    report_without_hash.pop("report_content_sha256", None)
    actual_report_hash = hashlib.sha256(
        json.dumps(report_without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_report_hash != expected_report_hash:
        raise ValueError("evaluation artifact hash mismatch")
    try:
        event_day = date.fromisoformat(snapshot.get("event_date", ""))
        trained_through = date.fromisoformat(final_fit["trained_through"])
    except (TypeError, ValueError, KeyError):
        raise ValueError("invalid prediction or training cutoff date") from None
    if event_day <= trained_through:
        raise ValueError("prediction event must be strictly later than the final training event")
    if snapshot.get("exploratory_only") is not True:
        raise ValueError("prediction snapshot must declare exploratory_only=true")
    for source in snapshot.get("candidate_sources", []):
        try:
            published = date.fromisoformat(str(source.get("published_at", ""))[:10])
        except (TypeError, ValueError):
            continue
        if published >= event_day:
            raise ValueError("same-day/future headline in prediction snapshot")
    temperature = float(final_fit["temperature"])
    ranked = _predict(
        final_fit["model"],
        _features(
            snapshot,
            headline_limit=final_fit.get("headline_limit"),
            selection=final_fit.get("headline_selection", "latest"),
        ),
    )
    label_names = final_fit["model"].get("label_names", {})
    return [
        {"rank": index, "idea_id": label, "idea": label_names.get(label, label),
         "score": round(score / temperature, 6), "score_type": "uncalibrated_logit"}
        for index, (label, score) in enumerate(ranked[:top_k], 1)
    ]


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
    config_grid = [
        {"alpha": alpha, "min_df": min_df, "headline_limit": headline_limit,
         "headline_selection": headline_selection, "temperature": temperature}
        for alpha in (0.5, 4.0)
        for min_df in (1, 3)
        for headline_limit in (20, 50, None)
        for headline_selection in ("latest", "spread_week")
        for temperature in (1.0, 8.0, 32.0)
    ]
    validation_rows = []
    for config in config_grid:
        model = _fit(
            [_features(snap_by_id[row["event_id"]], headline_limit=config["headline_limit"], selection=config["headline_selection"]) for row in train],
            [row["targets"] for row in train],
            alpha=config["alpha"], min_df=config["min_df"],
        )
        validation_labels = set(model["models"])
        result = [
            _metrics(
                [(label, score / config["temperature"]) for label, score in _predict(
                    model, _features(snap_by_id[row["event_id"]], headline_limit=config["headline_limit"], selection=config["headline_selection"])
                )],
                row["targets"], all_labels=validation_labels,
            )
            for row in validation
        ]
        validation_rows.append({"config": config, "metrics": _aggregate([{"metrics": item} for item in result])})
    selected = min(validation_rows, key=lambda row: (
        row["metrics"].get("binary_log_loss", float("inf")),
        -row["metrics"].get("ndcg_at_10", 0.0),
    ))
    model_config = selected["config"]
    model_fit_config = {key: model_config[key] for key in ("alpha", "min_df")}
    temperature = model_config["temperature"]
    headline_limit = model_config["headline_limit"]
    headline_selection = model_config["headline_selection"]
    # Refit on all pre-test examples; the test predictions remain chronological.
    # Incremental walk-forward refit: each target is scored only from examples dated earlier.
    test_rows = []
    for target in test:
        history = [row for row in train + validation + test if row["date"] < target["date"]]
        if not history:
            continue
        model_at_date = _fit(
            [_features(snap_by_id[row["event_id"]], headline_limit=headline_limit, selection=headline_selection) for row in history],
            [row["targets"] for row in history], **model_fit_config,
        )
        ranking = [(label, score / temperature) for label, score in _predict(model_at_date, _features(snap_by_id[target["event_id"]], headline_limit=headline_limit, selection=headline_selection))]
        metrics = _metrics(ranking, target["targets"], all_labels=set(model_at_date["models"]))
        test_rows.append({
            "event_id": target["event_id"], "event_date": target["date"].isoformat(),
            "retrieval_profile": target["profile"], "headline_count": target["source_count"],
            "target_idea_count": len(target["targets"]), "metrics": metrics,
            "target_ideas": sorted(target["targets"]),
            "top_10_predictions": [{"rank": rank, "idea_id": label,
                                   "idea": _label_names(history).get(label, label),
                                   "score": round(score, 6),
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
                              "metrics": _metrics(ranked, target["targets"], all_labels=set(vocabulary))})
    final_model = _fit(
        [_features(snap_by_id[row["event_id"]], headline_limit=headline_limit, selection=headline_selection) for row in examples],
        [row["targets"] for row in examples], **model_fit_config,
    )
    final_model["label_names"] = _label_names(examples)
    report: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "retrospective-context-to-idea-classifier-evaluation",
        "model_version": "bernoulli-headline-issue-classifier-v1",
        "task": "predict target-speech issue-phrase labels from dated news headlines published during the seven days before the target event; select the supervised model by binary cross-entropy loss then rank quality",
        "features": "headline title unigram/bigram presence from deduplicated, publisher-suffix-stripped titles in the preceding seven days; validation selects headline cap and either recent-only or day-stratified sampling. No target transcript, prior speech text, event-title metadata, future context, or history count is an inference feature",
        "supervised_training": "earlier examples use their own preceding-week context headlines as features and their own transcript-derived ideas as labels; only examples earlier than each scored event are fit",
        "target_policy": "compact transcript-extracted lexical issue phrases; weak, incomplete labels rather than semantic gold",
        "context_availability": "snapshots were retrieved retrospectively; feed publication dates do not establish actual pre-event availability",
        "snapshot_sha256": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
        "event_corpus_sha256": hashlib.sha256(Path(events_path).read_bytes()).hexdigest(),
        "target_extractor": {"version": ISSUE_PHRASE_EXTRACTOR_VERSION, "lexicon_sha256": issue_phrase_lexicon_sha256()},
        "example_count": len(examples), "skipped_snapshot_count": len(snapshots)-len(examples),
        "split": {"train_event_ids":[row["event_id"] for row in train],"validation_event_ids":[row["event_id"] for row in validation],"test_event_ids":[row["event_id"] for row in test],
                  "train_dates":[min(train_dates).isoformat(),max(train_dates).isoformat()],"validation_dates":[min(val_dates).isoformat(),max(val_dates).isoformat()],"test_dates":[min(test_dates).isoformat(),max(test_dates).isoformat()]},
        "selection_policy": "select Bernoulli smoothing, minimum headline document frequency, and logit temperature by mean validation binary log loss over labels known from earlier training events; target labels unseen in training are explicit misses with probability floor 1e-12; break ties by nDCG@10; refit for each target using earlier examples only",
        "loss_definition": "mean binary cross-entropy across idea IDs observed in training before each target; known negatives are included and target ideas absent from that training vocabulary receive the clipped probability floor, so novel ideas are penalized rather than omitted from the loss denominator",
        "validation_grid": validation_rows, "selected_config": model_config,
        "test_metrics": {"headline_conditioned_classifier": _aggregate(test_rows), "training_label_prevalence_reference": _aggregate(baseline_rows)},
        "test_predictions": test_rows,
        "final_fit": {"fit_event_count":len(examples),"training_target_label_count":sum(len(row["targets"]) for row in examples),
                       "candidate_label_count":len(final_model["models"]),"vocabulary_size":len(final_model["vocabulary"]),
                       "model_sha256":hashlib.sha256(json.dumps(final_model,sort_keys=True,separators=(",",":")).encode()).hexdigest()},
        "final_model": {"trained_through":max(row["date"] for row in examples).isoformat(),
                        "temperature":temperature,"headline_limit":headline_limit,
                        "headline_selection":headline_selection,"model":final_model},
        "limitations": ["Only 38 dated context/speech pairs are available in the merged snapshot set; chronological test results are tiny and unstable.",
                        "Binary loss is defined over labels observed in earlier training events; unseen target ideas are explicitly penalized at the clipped probability floor, but the incomplete phrase ontology still limits interpretation.",
                        "Binary loss is defined over labels observed in earlier training events; unseen target ideas are explicitly penalized at the clipped probability floor, but the incomplete phrase ontology still limits interpretation.",
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
