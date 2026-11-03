"""Time-coded event transcripts and manually reviewed response links."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


TIMECODE = re.compile(r"^(?:\d{2}:)?\d{2}:\d{2}$")
TURN_HEADER = re.compile(
    r"^(?P<speaker>.+?)\s+(?P<start>(?:\d{2}:)?\d{2}:\d{2})-"
    r"(?P<end>(?:\d{2}:)?\d{2}:\d{2})\s+\((?P<duration>[^)]*)\)$"
)
SCORE_ANNOTATION = re.compile(r"^(?P<label>.+?)\s+\((?P<score>-?\d+(?:\.\d+)?)\)$")


def parse_timecode(value: str) -> int:
    """Convert MM:SS or HH:MM:SS into whole seconds."""
    if not TIMECODE.fullmatch(value):
        raise ValueError(f"invalid timecode: {value!r}")
    parts = [int(part) for part in value.split(":")]
    if parts[-1] >= 60 or (len(parts) == 3 and parts[-2] >= 60):
        raise ValueError(f"invalid timecode: {value!r}")
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def _annotation(line: str) -> dict | None:
    if line.casefold() == "no stresslens":
        return {"source": "StressLens", "status": "unavailable", "raw": line}
    match = SCORE_ANNOTATION.fullmatch(line)
    if match:
        return {
            "source": "StressLens",
            "status": "scored",
            "label": match.group("label"),
            "score": float(match.group("score")),
            "raw": line,
        }
    return None


def parse_timestamped_transcript(
    text: str, metadata: dict, *, allow_timestamp_anomalies: bool = False,
) -> dict:
    """Parse a page's Full Transcript section without inferring response links."""
    if not isinstance(metadata, dict):
        raise ValueError("event metadata must be a JSON object")
    required = ("event_id", "event_date", "title", "source_url", "target_speaker")
    missing = [field for field in required if not isinstance(metadata.get(field), str) or not metadata[field].strip()]
    if missing:
        raise ValueError(f"missing event metadata: {', '.join(missing)}")
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", metadata["event_date"]):
            raise ValueError
        date.fromisoformat(metadata["event_date"])
    except ValueError:
        raise ValueError("event_date must use YYYY-MM-DD") from None

    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip().casefold() == "full transcript") + 1
    except StopIteration:
        raise ValueError("transcript page has no Full Transcript section") from None

    page_metadata = metadata.get("page_metadata", {})
    if not isinstance(page_metadata, dict):
        raise ValueError("page_metadata must be an object")
    for field in ("source_name", "reuse_terms"):
        if field in metadata and not isinstance(metadata[field], str):
            raise ValueError(f"{field} must be a string")
    aliases = metadata.get("target_speaker_aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
        raise ValueError("target_speaker_aliases must be a list of names")
    target_names = {metadata["target_speaker"].strip().casefold(), *(alias.strip().casefold() for alias in aliases)}
    turns = []
    timestamp_anomalies = []
    current = None
    body: list[str] = []
    last_start = -1

    def finish():
        if current is None:
            return
        raw_text = "\n".join(body).strip()
        turns.append({
            **current,
            "raw_text": raw_text,
            "text": " ".join(line.strip() for line in body if line.strip()),
        })

    for line in lines[start:]:
        stripped = line.strip()
        match = TURN_HEADER.fullmatch(stripped)
        if match:
            finish()
            begin = parse_timecode(match.group("start"))
            end = parse_timecode(match.group("end"))
            anomaly_reasons = []
            if end < begin:
                anomaly_reasons.append("negative_duration")
            if begin < last_start:
                anomaly_reasons.append("start_time_out_of_order")
            if anomaly_reasons and not allow_timestamp_anomalies:
                raise ValueError("turn timestamps must be ordered and have nonnegative duration")
            last_start = begin
            speaker_name = match.group("speaker").strip()
            normalized_speaker = speaker_name.casefold()
            role = (
                "questioner" if normalized_speaker == "question"
                else "target_speaker" if normalized_speaker in target_names
                else "other_speaker"
            )
            current = {
                "turn_id": f"{metadata['event_id']}:turn:{len(turns):04d}",
                "sequence": len(turns),
                "start_timecode": match.group("start"),
                "end_timecode": match.group("end"),
                "start_seconds": begin,
                "end_seconds": end,
                "speaker_name": speaker_name,
                "speaker_role": role,
                "timestamp_valid": not anomaly_reasons,
                "duration_label": match.group("duration").strip(),
                "annotations": [],
            }
            if anomaly_reasons:
                timestamp_anomalies.append({
                    "turn_id": current["turn_id"],
                    "start_timecode": match.group("start"),
                    "end_timecode": match.group("end"),
                    "reasons": anomaly_reasons,
                })
            body = []
            continue
        if current is None or not stripped:
            continue
        annotation = _annotation(stripped)
        if annotation:
            current["annotations"].append(annotation)
        else:
            body.append(line)
    finish()
    if not turns:
        raise ValueError("Full Transcript section contains no timestamped turns")

    return {
        "schema_version": 1,
        "event_id": metadata["event_id"],
        "event_date": metadata["event_date"],
        "title": metadata["title"],
        "source_url": metadata["source_url"],
        "source_name": metadata.get("source_name", ""),
        "reuse_terms": metadata.get("reuse_terms", ""),
        "target_speaker": metadata["target_speaker"],
        "target_speaker_aliases": aliases,
        "page_metadata": {**page_metadata, "raw_header": "\n".join(lines[:start - 1]).strip()},
        "turns": turns,
        "timestamp_anomalies": timestamp_anomalies,
        "prediction_examples": [],
    }


def add_prediction_examples(event: dict, annotations: Sequence[dict]) -> dict:
    """Attach manually reviewed trigger/response alignments and causal snapshots."""
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise ValueError("prediction annotations must be a list")
    if any(not isinstance(annotation, dict) for annotation in annotations):
        raise ValueError("each prediction annotation must be an object")
    if event.get("timestamp_anomalies") or any(
        not turn.get("timestamp_valid", True) for turn in event.get("turns", [])
    ):
        raise ValueError("cannot add response alignments to an event with timestamp anomalies")
    turns = event.get("turns", [])
    by_id = {turn["turn_id"]: turn for turn in turns}
    if len(by_id) != len(turns):
        raise ValueError("turn IDs must be unique")
    examples = []
    for index, annotation in enumerate(annotations):
        trigger_ids = annotation.get("trigger_turn_ids", [])
        target_ids = annotation.get("target_turn_ids", [])
        outcome_ids = annotation.get("outcome_turn_ids", [])
        for name, values in (("trigger_turn_ids", trigger_ids), ("target_turn_ids", target_ids),
                             ("outcome_turn_ids", outcome_ids)):
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"{name} must be a list of turn IDs")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        cutoff = annotation.get("prediction_time_seconds")
        status = annotation.get("response_status")
        if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 0:
            raise ValueError("prediction_time_seconds must be a nonnegative integer")
        if not trigger_ids or any(turn_id not in by_id for turn_id in trigger_ids):
            raise ValueError("each example must reference known trigger turns")
        turn_positions = {turn["turn_id"]: position for position, turn in enumerate(turns)}
        for name, ids in (("trigger_turn_ids", trigger_ids), ("target_turn_ids", target_ids),
                          ("outcome_turn_ids", outcome_ids)):
            if [turn_positions[turn_id] for turn_id in ids if turn_id in turn_positions] != sorted(
                turn_positions[turn_id] for turn_id in ids if turn_id in turn_positions
            ):
                raise ValueError(f"{name} must be in transcript order")
        if any(by_id[turn_id]["end_seconds"] > cutoff for turn_id in trigger_ids):
            raise ValueError("trigger turns must be fully observed by the prediction time")
        if not isinstance(status, str) or status not in {
            "target_spoke", "other_speaker_spoke", "no_target_response", "uncertain",
        }:
            raise ValueError("unsupported response_status")
        if (status == "target_spoke") != bool(target_ids):
            raise ValueError("target_spoke examples require target_turn_ids, and other outcomes must omit them")
        if any(turn_id not in by_id for turn_id in [*target_ids, *outcome_ids]):
            raise ValueError("response links must reference known turns")
        if any(by_id[turn_id]["speaker_role"] != "target_speaker" for turn_id in target_ids):
            raise ValueError("target_turn_ids must belong to the configured target speaker")
        if set(target_ids) & set(outcome_ids):
            raise ValueError("target and outcome turn IDs must be distinct")
        if status == "other_speaker_spoke" and not outcome_ids:
            raise ValueError("other_speaker_spoke examples require outcome_turn_ids")
        if status == "other_speaker_spoke" and any(
            by_id[turn_id]["speaker_role"] == "target_speaker" for turn_id in outcome_ids
        ):
            raise ValueError("other-speaker outcomes cannot identify target-speaker turns")
        if any(by_id[turn_id]["start_seconds"] < cutoff for turn_id in [*target_ids, *outcome_ids]):
            raise ValueError("response turns must start at or after the prediction time")
        idea_labels = annotation.get("idea_labels", [])
        annotation_source = annotation.get("annotation_source", "")
        response_kind = annotation.get("response_kind", "")
        if not isinstance(idea_labels, list) or any(not isinstance(label, str) or not label for label in idea_labels):
            raise ValueError("idea_labels must be a list of nonempty strings")
        if not isinstance(annotation_source, str) or not annotation_source.strip():
            raise ValueError("annotation_source is required")
        if not isinstance(response_kind, str):
            raise ValueError("response_kind must be a string")
        excluded = set(target_ids) | set(outcome_ids)
        visible = [
            turn["turn_id"] for turn in turns
            if (
                turn["start_seconds"] < cutoff
                and turn["end_seconds"] <= cutoff
                and turn["turn_id"] not in excluded
            )
        ]
        if not set(trigger_ids).issubset(visible):
            raise ValueError("trigger turns must be visible at the prediction point")
        examples.append({
            "example_id": annotation.get("example_id", f"{event['event_id']}:prediction:{index:04d}"),
            "prediction_time_seconds": cutoff,
            "trigger_turn_ids": list(trigger_ids),
            "visible_turn_ids": visible,
            "target_turn_ids": list(target_ids),
            "outcome_turn_ids": list(outcome_ids),
            "response_status": status,
            "response_kind": response_kind,
            "idea_labels": list(idea_labels),
            "annotation_source": annotation_source,
        })
    return {**event, "prediction_examples": examples}


def prediction_example_inputs(event: dict, example: dict) -> dict:
    """Materialize one causal model input from its stored turn IDs."""
    if event.get("timestamp_anomalies") or any(
        not turn.get("timestamp_valid", True) for turn in event.get("turns", [])
    ):
        raise ValueError("cannot materialize model inputs from an event with timestamp anomalies")
    by_id = {turn["turn_id"]: turn for turn in event.get("turns", [])}
    visible = example.get("visible_turn_ids", [])
    triggers = example.get("trigger_turn_ids", [])
    targets = example.get("target_turn_ids", [])
    for name, ids in (("visible_turn_ids", visible), ("trigger_turn_ids", triggers),
                      ("target_turn_ids", targets)):
        if not isinstance(ids, list) or any(not isinstance(turn_id, str) for turn_id in ids):
            raise ValueError(f"{name} must be a list of turn IDs")
    if not triggers:
        raise ValueError("prediction example must identify at least one trigger turn")
    if any(turn_id not in by_id for turn_id in [*visible, *triggers, *targets]):
        raise ValueError("prediction example references unknown turns")
    cutoff = example.get("prediction_time_seconds")
    if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 0:
        raise ValueError("prediction_time_seconds must be a nonnegative integer")
    if any(
        by_id[turn_id]["start_seconds"] >= cutoff or by_id[turn_id]["end_seconds"] > cutoff
        for turn_id in visible
    ):
        raise ValueError("visible turns must be fully observed before the prediction point")
    if not set(triggers).issubset(visible):
        raise ValueError("trigger turns must be visible at the prediction point")
    if any(
        by_id[turn_id]["speaker_role"] != "target_speaker"
        or by_id[turn_id]["start_seconds"] < cutoff
        for turn_id in targets
    ):
        raise ValueError("target turns must be target-speaker text after the prediction point")
    if set(targets) & set(visible):
        raise ValueError("target text must not appear in the observed context")
    trigger_set = set(triggers)
    prefix = [turn_id for turn_id in visible if turn_id not in trigger_set]

    def model_turn(turn_id):
        turn = by_id[turn_id]
        return {key: turn[key] for key in (
            "turn_id", "sequence", "start_seconds", "end_seconds",
            "speaker_name", "speaker_role", "text",
        )}

    return {
        "event_id": event["event_id"],
        "example_id": example["example_id"],
        "prediction_time_seconds": cutoff,
        "input": {
            "event_date": event["event_date"],
            "target_speaker": event["target_speaker"],
            "observed_turns": [model_turn(turn_id) for turn_id in visible],
            "trigger_turns": [model_turn(turn_id) for turn_id in triggers],
            "prefix_text": " ".join(
                f"{by_id[turn_id]['speaker_name']}: {by_id[turn_id]['text']}" for turn_id in prefix
            ),
            "question_text": " ".join(
                f"{by_id[turn_id]['speaker_name']}: {by_id[turn_id]['text']}" for turn_id in triggers
            ),
        },
        "target": {
            "turn_ids": list(targets),
            "text": " ".join(by_id[turn_id]["text"] for turn_id in targets),
            "response_status": example["response_status"],
            "response_kind": example["response_kind"],
            "idea_labels": list(example["idea_labels"]),
        },
        "provenance": {
            "title": event["title"],
            "source_url": event["source_url"],
            "source_name": event["source_name"],
            "reuse_terms": event["reuse_terms"],
            "page_metadata": event["page_metadata"],
            "turn_annotations": {
                turn["turn_id"]: turn["annotations"] for turn in event["turns"]
            },
        },
    }


def append_events(path: str | Path, events: Iterable[dict]) -> int:
    """Append event rows, skipping IDs already present in the JSONL file."""
    output = Path(path)
    existing_ids = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing_ids.add(json.loads(line)["event_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("a", encoding="utf-8") as stream:
        for event in events:
            event_id = event["event_id"]
            if event_id in existing_ids:
                continue
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            existing_ids.add(event_id)
            written += 1
    return written


def append_event(path: str | Path, event: dict) -> None:
    """Append one event JSONL row, refusing duplicate event IDs."""
    if append_events(path, [event]) == 0:
        raise ValueError(f"event already exists: {event['event_id']}")
