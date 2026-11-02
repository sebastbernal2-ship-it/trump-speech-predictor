"""Conservative paragraph segmentation and speaker/turn classification."""

from __future__ import annotations

import re
from typing import Iterable

from .audit import classify_title


SPEAKER_PATTERNS = (
    ("questioner", re.compile(r"^\s*(Q(?:UESTION)?)\s*[.:—-]\s*", re.IGNORECASE)),
    ("president", re.compile(r"^\s*(THE PRESIDENT|PRESIDENT TRUMP|DONALD J\. TRUMP|TRUMP)\s*[.:—-]\s*", re.IGNORECASE)),
    ("other", re.compile(r"^\s*((?:MR|MS|MRS|DR)\.\s+[^.\n]{1,60})\.\s*", re.IGNORECASE)),
)
CLASSIFIER_VERSION = "explicit-speaker-prefix-v1"


def segment_record(record: dict) -> list[dict]:
    """Split on transcript paragraph boundaries; infer roles only from explicit labels."""
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", record.get("text", ""))
        if paragraph.strip()
    ]
    segments = []
    active_question_id = None
    answer_started = False
    record_type = record.get("provisional_type") or classify_title(record.get("event", ""))
    for index, paragraph in enumerate(paragraphs):
        role, label, content = "unknown", None, paragraph
        for candidate_role, pattern in SPEAKER_PATTERNS:
            match = pattern.match(paragraph)
            if match:
                role = candidate_role
                label = match.group(1).strip()
                content = paragraph[match.end() :].strip()
                break

        segment_id = f"{record['speech_id']}:{index:04d}"
        response_to = None
        if role == "questioner":
            kind = "question"
            active_question_id = segment_id
            answer_started = False
            confidence = 0.99
        elif role == "president":
            kind = "presidential_answer" if active_question_id else "presidential_remarks"
            response_to = active_question_id
            answer_started = bool(response_to)
            confidence = 0.95 if response_to else 0.90
        elif role == "other":
            kind = "other_speaker_turn"
            confidence = 0.95
        else:
            kind = "unattributed_text"
            confidence = 0.0

        if role not in {"questioner", "president"} and answer_started:
            active_question_id = None
            answer_started = False

        segments.append({
            "segment_id": segment_id,
            "speech_id": record["speech_id"],
            "date": record["date"],
            "event": record["event"],
            "source": record["source"],
            "record_type": record_type,
            "segment_index": index,
            "speaker_role": role,
            "speaker_label": label,
            "segment_type": kind,
            "response_to": response_to,
            "label_method": "explicit_speaker_prefix" if role != "unknown" else "unlabeled",
            "label_confidence": confidence,
            "text": paragraph,
            "utterance_text": content,
            "classifier_version": CLASSIFIER_VERSION,
        })
    return segments


def segment_records(records: Iterable[dict]) -> Iterable[dict]:
    for record in records:
        yield from segment_record(record)
