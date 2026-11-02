"""Create disjoint, task-oriented views of the Roll Call event corpus."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile


SPEECH_RECORD_TYPES = {"Speech", "Remarks", "Weekly Address", "Convention"}
OUTPUT_FILES = {
    "qa_candidates": "trump-qa-candidates.jsonl",
    "speech_candidates": "trump-speech-candidates.jsonl",
    "other_trump_events": "trump-other-events.jsonl",
    "non_trump_events": "non-trump-events.jsonl",
}


def _event_group(event: dict) -> str:
    turns = event.get("turns", [])
    has_trump = any(turn.get("speaker_role") == "target_speaker" for turn in turns)
    has_questioner = any(turn.get("speaker_role") == "questioner" for turn in turns)
    if not has_trump:
        return "non_trump_events"
    if has_questioner:
        return "qa_candidates"
    page_metadata = event.get("page_metadata") or {}
    record = page_metadata.get("factbase_record") or {}
    if record.get("record_type") in SPEECH_RECORD_TYPES:
        return "speech_candidates"
    return "other_trump_events"


def extract_long_speech_records(
    input_path: str | Path = "data/rollcall-by-type/trump-speech-candidates.jsonl",
    output_path: str | Path = "data/rollcall-by-type/trump-long-speeches.jsonl",
    *,
    min_target_words: int = 1000,
    min_target_share: float = 0.8,
) -> dict[str, int]:
    """Export target-only, long speech candidates in the legacy training schema."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if min_target_words < 1 or not 0 <= min_target_share <= 1:
        raise ValueError("minimum words must be positive and target share must be between 0 and 1")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing speech dataset: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}-", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        selected = 0
        try:
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    turns = event.get("turns", [])
                    target_turns = [
                        turn for turn in turns
                        if turn.get("speaker_role") == "target_speaker" and turn.get("text", "").strip()
                    ]
                    if not target_turns or any(
                        turn.get("speaker_role") == "questioner" for turn in turns
                    ):
                        continue
                    source_record = (event.get("page_metadata") or {}).get("factbase_record") or {}
                    record_type = source_record.get("record_type")
                    if record_type not in SPEECH_RECORD_TYPES:
                        continue
                    target_text = "\n\n".join(turn["text"].strip() for turn in target_turns)
                    target_words = len(target_text.split())
                    total_words = sum(len(turn.get("text", "").split()) for turn in turns)
                    share = target_words / total_words if total_words else 0.0
                    if target_words < min_target_words or share < min_target_share:
                        continue
                    row = {
                        "speech_id": event["event_id"],
                        "date": event["event_date"],
                        "event": event["title"],
                        "source": event["source_url"],
                        "text": target_text,
                        "source_record_type": record_type,
                        "target_word_count": target_words,
                        "target_word_share": round(share, 4),
                    }
                    temporary.write(json.dumps(row, ensure_ascii=False) + "\n")
                    selected += 1
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            if isinstance(error, json.JSONDecodeError):
                raise ValueError(f"invalid JSON in {source} near line {line_number}") from error
            raise
    try:
        os.link(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.unlink()
    return {"selected_events": selected}


def split_rollcall_events(
    input_path: str | Path = "data/rollcall-events.jsonl",
    output_dir: str | Path = "data/rollcall-by-type",
) -> dict[str, int]:
    """Write a disjoint event-level task split without modifying the source corpus.

    Q&A candidates contain both target-speaker and questioner turns; this does
    not claim that any particular question was answered by any particular turn.
    """
    source = Path(input_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing split directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    counts: Counter[str] = Counter()
    try:
        with ExitStack() as stack:
            outputs = {
                group: stack.enter_context((temporary / name).open("w", encoding="utf-8"))
                for group, name in OUTPUT_FILES.items()
            }
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON on line {line_number}") from error
                    if not isinstance(event, dict) or not isinstance(event.get("turns"), list):
                        raise ValueError(f"invalid event structure on line {line_number}")
                    group = _event_group(event)
                    outputs[group].write(line if line.endswith("\n") else line + "\n")
                    counts[group] += 1
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {group: counts[group] for group in OUTPUT_FILES}
