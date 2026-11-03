"""Diagnostics for in-speech hierarchical plan-path predictions."""

from __future__ import annotations

from collections import Counter
import math
import statistics
import time
import torch

from .explicit_plan import (
    _chronological_split,
    _encode_text,
    _event_records,
    load_explicit_plan_model,
    parse_plan_path,
)
from .model import tokenize
from .speech_plans import plan_path_examples
from .weak_plan_labels import _words


LEVELS = ("theme", "section", "point")


def _normalized(text: str) -> str:
    return " ".join(tokenize(text))


def _lexical_tokens(text: str) -> set[str]:
    return set(_words(text))


def _accumulator() -> dict:
    return {"examples": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0, "macro_f1": 0.0}


def _add_score(accumulator: dict, prediction: str, reference: str) -> None:
    predicted, expected = _lexical_tokens(prediction), _lexical_tokens(reference)
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accumulator["examples"] += 1
    accumulator["exact"] += _normalized(prediction) == _normalized(reference)
    accumulator["tp"] += true_positive
    accumulator["fp"] += false_positive
    accumulator["fn"] += false_negative
    accumulator["macro_f1"] += f1


def _finish(accumulator: dict) -> dict[str, float | int]:
    tp, fp, fn = accumulator["tp"], accumulator["fp"], accumulator["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "examples": accumulator["examples"],
        "exact_match": accumulator["exact"] / max(accumulator["examples"], 1),
        "token_micro_precision": precision,
        "token_micro_recall": recall,
        "token_micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "token_macro_f1": accumulator["macro_f1"] / max(accumulator["examples"], 1),
    }


def _top_terms(counter: Counter[str], count: int) -> str:
    return " ".join(word for word, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:count])


def _progress_bin(event: dict, cutoff: int) -> str:
    timed = [
        turn for turn in event.get("turns", [])
        if type(turn.get("start_seconds")) is int and type(turn.get("end_seconds")) is int
    ]
    if not timed:
        return "unknown"
    start = min(turn["start_seconds"] for turn in timed)
    end = max(turn["end_seconds"] for turn in timed)
    fraction = (cutoff - start) / max(end - start, 1)
    return "early" if fraction < 0.25 else "mid_early" if fraction < 0.5 else "mid_late" if fraction < 0.75 else "late"


def evaluate_explicit_plan_model(
    checkpoint_path: str,
    events_path: str,
    plans_path: str,
    *,
    device_name: str = "auto",
    cpu_threads: int | None = None,
) -> dict:
    """Compare held-out hierarchical predictions with weak-label and causal lexical baselines."""
    model, vocabulary, device, metadata = load_explicit_plan_model(checkpoint_path, device_name)
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if device.type == "cpu":
            torch.set_num_threads(cpu_threads)
    version = metadata.get("model_version", {})
    label_policy = version.get("training", {}).get("label_policy", "reviewed")
    records = _event_records(events_path, plans_path, label_policy)
    train_records, validation_records, test_records = _chronological_split(records)
    train_examples = [
        example for event, plan in train_records
        for example in plan_path_examples(
            event, plan, max_context_tokens=model.config.context_length, label_policy=label_policy,
        )
    ]
    test_examples = [
        example for event, plan in test_records
        for example in plan_path_examples(
            event, plan, max_context_tokens=model.config.context_length, label_policy=label_policy,
        )
    ]
    if not train_examples or not test_examples:
        raise ValueError("training and test splits must both contain plan-path examples")

    theme_counts: Counter[str] = Counter()
    keyword_counts = {"section": Counter(), "point": Counter()}
    for event, plan in train_records:
        for theme in plan["themes"]:
            theme_counts[_normalized(theme["summary"])] += 1
    for example in train_examples:
        for level in keyword_counts:
            keyword_counts[level].update(_words(example["target"][level]))
    majority_theme = min(theme_counts, key=lambda label: (-theme_counts[label], label))

    event_by_id = {event["event_id"]: event for event, _ in test_records}
    accumulators = {
        group: {level: _accumulator() for level in LEVELS}
        for group in ("overall", "early", "mid_early", "mid_late", "late")
    }
    baselines = {
        name: {level: _accumulator() for level in LEVELS}
        for name in ("majority_theme", "training_keywords", "recent_prefix_keywords")
    }
    valid_paths = 0
    latencies = []
    repetition_ratios = []
    prediction_counts = {level: Counter() for level in LEVELS}
    reference_counts = {level: Counter() for level in LEVELS}
    lookup = {token: index for index, token in enumerate(vocabulary)}
    recent_prefix_counter = Counter()
    for example in test_examples:
        context = _encode_text(example["prefix_text"], lookup, model.config.context_length)
        context_ids = torch.tensor([context], dtype=torch.long, device=device)
        started = time.perf_counter()
        generated = model.generate(context_ids, max_new_tokens=model.config.plan_length)
        latencies.append((time.perf_counter() - started) * 1000)
        words = [vocabulary[index] for index in generated if 0 <= index < len(vocabulary)]
        try:
            prediction = parse_plan_path(words)
            valid_paths += 1
        except ValueError:
            prediction = {level: "" for level in LEVELS}
        event = event_by_id[example["event_id"]]
        group = _progress_bin(event, example["prediction_time_seconds"])
        recent_prefix_counter.clear()
        recent_prefix_counter.update(_words(example["prefix_text"]))
        output_repetition = [word for level in LEVELS for word in _words(prediction[level])]
        repetition_ratios.append(
            1 - len(set(output_repetition)) / len(output_repetition) if output_repetition else 0.0
        )
        for level in LEVELS:
            reference = example["target"][level]
            prediction_counts[level][_normalized(prediction[level])] += 1
            reference_counts[level][_normalized(reference)] += 1
            _add_score(accumulators["overall"][level], prediction[level], reference)
            if group in accumulators:
                _add_score(accumulators[group][level], prediction[level], reference)
            if level == "theme":
                _add_score(baselines["majority_theme"][level], majority_theme, reference)
            else:
                count = max(1, len(_lexical_tokens(reference)))
                common = _top_terms(keyword_counts[level], count)
                recent = _top_terms(recent_prefix_counter, count)
                _add_score(baselines["training_keywords"][level], common, reference)
                _add_score(baselines["recent_prefix_keywords"][level], recent, reference)

    ordered_latencies = sorted(latencies)
    p95_index = min(len(ordered_latencies) - 1, math.ceil(0.95 * len(ordered_latencies)) - 1)
    return {
        "model_version": version.get("model_version"),
        "label_policy": label_policy,
        "device": str(device),
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "split": {
            "train_events": len(train_records),
            "validation_events": len(validation_records),
            "test_events": len(test_records),
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
            "test_date_range": [test_records[0][0]["event_date"], test_records[-1][0]["event_date"]],
        },
        "overall": {
            "valid_path_rate": valid_paths / len(test_examples),
            "latency_scope": "model.generate only; excludes loading, tokenization, and transcript streaming",
            "mean_generation_latency_ms": statistics.mean(latencies),
            "p50_generation_latency_ms": statistics.median(latencies),
            "p95_generation_latency_ms": ordered_latencies[p95_index],
            "mean_repetition_ratio": statistics.mean(repetition_ratios),
            "levels": {level: _finish(accumulators["overall"][level]) for level in LEVELS},
        },
        "prediction_diversity": {
            level: {
                "unique_predictions": len(prediction_counts[level]),
                "top_predictions": [
                    {"label": label, "count": count}
                    for label, count in prediction_counts[level].most_common(5)
                ],
                "unique_references": len(reference_counts[level]),
                "top_references": [
                    {"label": label, "count": count}
                    for label, count in reference_counts[level].most_common(5)
                ],
            }
            for level in LEVELS
        },
        "baselines": {
            name: {level: _finish(baselines[name][level]) for level in LEVELS if baselines[name][level]["examples"]}
            for name in baselines
        },
        "event_progress_bins": {
            "early": "0-25% of source event duration",
            "mid_early": "25-50%",
            "mid_late": "50-75%",
            "late": "75-100%",
        },
        "by_event_progress": {
            group: {
                level: _finish(accumulators[group][level])
                for level in LEVELS if accumulators[group][level]["examples"]
            }
            for group in ("early", "mid_early", "mid_late", "late")
        },
    }
