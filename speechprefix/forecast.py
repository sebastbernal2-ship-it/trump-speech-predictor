"""A simple pre-speech recurring-phrase baseline with a chronological protocol."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .model import tokenize


REQUIRED_FIELDS = {"speech_id", "date", "event", "source", "text"}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL speech records and require dates and provenance."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing fields: {', '.join(sorted(missing))}")
            if not all(isinstance(record[field], str) and record[field].strip() for field in REQUIRED_FIELDS):
                raise ValueError(f"line {line_number}: required fields must be non-empty strings")
            records.append(record)
    records.sort(key=lambda record: (record["date"], record["speech_id"]))
    if len({record["speech_id"] for record in records}) != len(records):
        raise ValueError("speech_id values must be unique")
    return records


def phrase_set(text: str, min_words: int = 2, max_words: int = 4) -> set[str]:
    """Extract normalized phrases, excluding common connectors and editorial notes."""
    text = re.sub(r"(?im)^\s*note:\s*the president spoke[^\n]*", " ", text)
    stop = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it", "of", "on", "or", "s", "that", "the", "to", "u", "was", "we", "were", "with", "you"}
    words = [word for word in tokenize(text) if re.fullmatch(r"[\w]+(?:['’][\w]+)*", word)]
    found = set()
    for size in range(min_words, max_words + 1):
        for start in range(len(words) - size + 1):
            phrase = words[start : start + size]
            if sum(word not in stop for word in phrase) >= 2:
                found.add(" ".join(phrase))
    return found


def forecast_phrases(history: list[dict[str, Any]], top: int = 20) -> list[tuple[str, float]]:
    """Rank phrases by the fraction of prior speeches in which each phrase appeared."""
    if top < 1:
        raise ValueError("top must be positive")
    if not history:
        raise ValueError("at least one historical speech is required")
    document_counts: Counter[str] = Counter()
    for record in history:
        document_counts.update(phrase_set(record["text"]))
    n_speeches = len(history)
    ranked = [(phrase, count / n_speeches) for phrase, count in document_counts.items()]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:top]


def evaluate_phrase_forecast(records: list[dict[str, Any]], min_history: int = 3) -> dict[str, float | int]:
    """Walk forward; score per-speech phrase occurrence probabilities from prior speech frequency."""
    if min_history < 1:
        raise ValueError("min_history must be positive")
    if len(records) <= min_history:
        raise ValueError("need more records than min_history")
    squared_error = 0.0
    outcomes = 0
    hits_at_10 = 0
    scored_speeches = 0
    for index in range(min_history, len(records)):
        history = records[:index]
        target = phrase_set(records[index]["text"])
        predictions = forecast_phrases(history, top=10)
        hits_at_10 += sum(phrase in target for phrase, _ in predictions)
        # Brier score over the candidate phrase vocabulary seen in history.
        candidates = set().union(*(phrase_set(record["text"]) for record in history))
        probabilities = dict(forecast_phrases(history, top=max(1, len(candidates))))
        for phrase in candidates:
            squared_error += (probabilities.get(phrase, 0.0) - float(phrase in target)) ** 2
            outcomes += 1
        scored_speeches += 1
    return {
        "speeches_scored": scored_speeches,
        "candidate_phrase_brier": squared_error / outcomes if outcomes else 0.0,
        "mean_hits_in_top10": hits_at_10 / scored_speeches,
    }
