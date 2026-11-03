"""Build explicitly weak extractive plan labels from local timestamped speeches."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .model import tokenize
from .speech_plans import WEAK_PLAN_ANNOTATION_SOURCE


STOP_WORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "even", "every", "few", "for", "from",
    "further", "get", "got", "had", "has", "have", "having", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "know", "let", "like", "may",
    "me", "more", "most", "much", "must", "my", "near", "no", "nor", "not", "now", "of", "off",
    "on", "once", "one", "only", "or", "other", "our", "out", "over", "own", "people", "said", "say",
    "says", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "us",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "would", "you", "your", "trump", "president", "uh", "um", "gonna", "wanna", "gotta",
    "thank", "thanks", "well", "okay", "ok", "right", "today", "tonight", "everybody", "everyone",
    "folks", "ladies", "gentlemen", "welcome", "congratulations", "honored", "appreciate", "grateful",
    "great", "good", "wonderful",
}


def _words(text: str) -> list[str]:
    return [word for word in tokenize(text) if word.isalpha() and len(word) >= 3 and word not in STOP_WORDS]


def _top_terms(text: str, idf: dict[str, float], limit: int) -> str:
    counts = Counter(_words(text))
    ranked = sorted(counts, key=lambda word: (-counts[word] * idf.get(word, 1.0), -counts[word], word))
    selected = ranked[:limit]
    if not selected:
        selected = [word for word in tokenize(text) if word.isalpha()][:limit]
    return " ".join(selected) or "general remarks"


def _load_selected_records(path: Path) -> dict[str, dict]:
    selected = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            speech_id, date = row.get("speech_id"), row.get("date")
            if not isinstance(speech_id, str) or speech_id in selected or not isinstance(date, str):
                raise ValueError(f"speech IDs and dates must be present and unique (line {line_number})")
            selected[speech_id] = row
    if not selected:
        raise ValueError("speech-record file is empty")
    return selected


def _target_turns(event: dict) -> list[dict[str, Any]]:
    turns = [
        turn for turn in event.get("turns", [])
        if turn.get("speaker_role") == "target_speaker" and turn.get("text", "").strip()
    ]
    return turns


def _document_frequency(events_path: Path, selected_ids: set[str]) -> tuple[Counter[str], int]:
    document_frequency: Counter[str] = Counter()
    seen = set()
    with events_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            event_id = event.get("event_id")
            if event_id not in selected_ids:
                continue
            if event_id in seen:
                raise ValueError(f"duplicate selected source event: {event_id}")
            seen.add(event_id)
            text = " ".join(turn["text"] for turn in _target_turns(event))
            document_frequency.update(set(_words(text)))
    missing = selected_ids - seen
    if missing:
        raise ValueError(f"selected source events missing from corpus: {', '.join(sorted(missing)[:5])}")
    return document_frequency, len(seen)


def _source_theme(event: dict) -> tuple[str | None, str | None]:
    record = (event.get("page_metadata") or {}).get("factbase_record") or {}
    topics = (record.get("document") or {}).get("topics") or []
    eligible = [
        topic for topic in topics
        if isinstance(topic, dict) and isinstance(topic.get("topic"), str)
        and topic["topic"].strip()
        and isinstance(topic.get("score"), (int, float))
        and not isinstance(topic.get("score"), bool)
        and math.isfinite(float(topic["score"]))
    ]
    if not eligible:
        return None, None
    selected = max(eligible, key=lambda topic: (float(topic["score"]), topic["topic"]))
    return selected["topic"].strip(), selected.get("source")


def _make_plan(
    event: dict,
    idf: dict[str, float],
    *,
    section_turns: int,
    point_turns: int,
    top_terms: int,
    idf_training_events: int,
    idf_training_date_range: list[str],
) -> dict:
    turns = _target_turns(event)
    if not turns:
        raise ValueError(f"selected event has no target-speaker transcript: {event.get('event_id')}")
    if event.get("timestamp_anomalies") or any(
        not turn.get("timestamp_valid", True)
        or type(turn.get("start_seconds")) is not int
        or type(turn.get("end_seconds")) is not int
        for turn in turns
    ):
        raise ValueError(f"selected event has invalid timestamps: {event.get('event_id')}")
    theme_summary, theme_source = _source_theme(event)
    if not theme_summary:
        theme_summary = _top_terms(" ".join(turn["text"] for turn in turns), idf, top_terms)
        theme_source = "transcript-keywords"
    theme_turn_ids = [turn["turn_id"] for turn in turns]
    sections = []
    for section_index, section_start in enumerate(range(0, len(turns), section_turns)):
        section = turns[section_start : section_start + section_turns]
        section_ids = [turn["turn_id"] for turn in section]
        section_summary = _top_terms(" ".join(turn["text"] for turn in section), idf, top_terms)
        points = []
        for point_index, point_start in enumerate(range(0, len(section), point_turns)):
            point = section[point_start : point_start + point_turns]
            points.append({
                "id": f"point-{section_index:04d}-{point_index:04d}",
                "summary": _top_terms(" ".join(turn["text"] for turn in point), idf, top_terms),
                "turn_ids": [turn["turn_id"] for turn in point],
            })
        sections.append({
            "id": f"section-{section_index:04d}",
            "summary": section_summary,
            "turn_ids": section_ids,
            "talking_points": points,
        })
    return {
        "schema_version": 1,
        "event_id": event["event_id"],
        "review_status": "draft",
        "reviewed_by": None,
        "annotation_source": WEAK_PLAN_ANNOTATION_SOURCE,
        "weak_label_config": {
            "method": "source topic for theme when available; tf-idf transcript keywords in fixed target-turn windows",
            "theme_source": theme_source,
            "section_turns": section_turns,
            "point_turns": point_turns,
            "top_terms": top_terms,
            "idf_training_events": idf_training_events,
            "idf_training_date_range": idf_training_date_range,
            "idf_policy": "chronological training-date partition only",
        },
        "themes": [{
            "id": "theme-0000",
            "summary": theme_summary,
            "turn_ids": theme_turn_ids,
            "sections": sections,
        }],
    }


def create_weak_plan_labels(
    events_path: str | Path,
    speech_records_path: str | Path,
    output_path: str | Path,
    *,
    section_turns: int = 15,
    point_turns: int = 3,
    top_terms: int = 4,
) -> dict[str, int | str]:
    """Write local pseudo-labels; these are not human-reviewed semantic plans."""
    if min(section_turns, point_turns, top_terms) < 1:
        raise ValueError("turn-window sizes and top_terms must be positive")
    source, records, destination = Path(events_path), Path(speech_records_path), Path(output_path)
    if not source.is_file() or not records.is_file():
        raise FileNotFoundError(source if not source.is_file() else records)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite weak plan labels: {destination}")
    selected_records = _load_selected_records(records)
    selected_ids = set(selected_records)
    dates = sorted({row["date"] for row in selected_records.values()})
    if len(dates) < 3:
        raise ValueError("weak plan labeling requires at least three distinct event dates")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * 0.8)))
    train_dates = set(dates[:train_end])
    idf_training_ids = {
        speech_id for speech_id, row in selected_records.items() if row["date"] in train_dates
    }
    document_frequency, document_count = _document_frequency(source, idf_training_ids)
    idf = {
        word: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for word, frequency in document_frequency.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}-", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        written = 0
        try:
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("event_id") not in selected_ids:
                        continue
                    speech_record = selected_records[event["event_id"]]
                    if event.get("event_date") != speech_record["date"]:
                        raise ValueError(f"speech date does not match source event: {event['event_id']}")
                    plan = _make_plan(
                        event, idf, section_turns=section_turns,
                        point_turns=point_turns, top_terms=top_terms,
                        idf_training_events=document_count,
                        idf_training_date_range=[min(train_dates), max(train_dates)],
                    )
                    temporary.write(json.dumps(plan, ensure_ascii=False) + "\n")
                    written += 1
            if written != len(selected_ids):
                raise ValueError(f"generated {written} plans for {len(selected_ids)} selected events")
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.unlink()
    return {
        "events": written,
        "plans": str(destination),
        "annotation_source": WEAK_PLAN_ANNOTATION_SOURCE,
    }
