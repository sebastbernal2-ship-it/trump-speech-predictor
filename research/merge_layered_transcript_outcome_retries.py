"""Merge validated retry rows over failed rows without modifying prior artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
BASE_DEFAULT = ROOT / "data/source-archives/2019-layered-transcript-outcomes-v1.jsonl"
RETRY_DEFAULT = ROOT / "data/source-archives/2019-layered-outcomes-retry-v1.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/2019-layered-transcript-outcomes-v1-complete.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _read(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    return rows, raw


def _target_turns(record: dict[str, Any]) -> list[dict[str, str]]:
    target = str(record.get("target_speaker", "")).casefold()
    turns = []
    for turn in record.get("turns", []):
        if not isinstance(turn, dict):
            continue
        is_target = turn.get("speaker_role") == "target_speaker" or (
            target and str(turn.get("speaker_name", "")).casefold() == target
        )
        text, turn_id = turn.get("text") or turn.get("raw_text"), turn.get("turn_id")
        if is_target and isinstance(text, str) and text.strip():
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError("target-speaker turn is missing a turn_id")
            turns.append({"turn_id": turn_id, "text": text.strip()})
    return turns


def merge(*, events_path: Path, base_path: Path, retry_paths: list[Path], output_path: Path) -> dict[str, Any]:
    if not retry_paths:
        raise ValueError("at least one retry artifact is required")
    events, event_bytes = _read(events_path)
    base_rows, base_bytes = _read(base_path)
    base_manifest_path = Path(str(base_path) + ".manifest.json")
    base_manifest_bytes = base_manifest_path.read_bytes()
    base_manifest = json.loads(base_manifest_bytes)
    if base_manifest.get("output_sha256") != _sha256(base_bytes):
        raise ValueError("base layered artifact hash mismatch")
    if base_manifest.get("events_sha256") != _sha256(event_bytes):
        raise ValueError("event corpus differs from the base extraction run")
    event_by_id = {row.get("event_id"): row for row in events}
    base_by_id = {row.get("event_id"): row for row in base_rows}
    if len(event_by_id) != len(events) or len(base_by_id) != len(base_rows):
        raise ValueError("event IDs must be unique")
    retry_by_id: dict[str, dict[str, Any]] = {}
    retry_artifact_hashes, retry_manifest_hashes = [], []
    for retry_path in retry_paths:
        retry_rows, retry_bytes = _read(retry_path)
        retry_manifest_bytes = Path(str(retry_path) + ".manifest.json").read_bytes()
        retry_manifest = json.loads(retry_manifest_bytes)
        if retry_manifest.get("output_sha256") != _sha256(retry_bytes):
            raise ValueError(f"retry layered artifact hash mismatch: {retry_path}")
        if retry_manifest.get("events_sha256") != _sha256(event_bytes):
            raise ValueError(f"event corpus differs from retry extraction run: {retry_path}")
        row_ids = [row.get("event_id") for row in retry_rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"retry event IDs must be unique: {retry_path}")
        retry_artifact_hashes.append(_sha256(retry_bytes))
        retry_manifest_hashes.append(_sha256(retry_manifest_bytes))
        for row in retry_rows:
            event_id = row.get("event_id")
            existing = retry_by_id.get(event_id)
            if existing is None or (existing.get("status") != "validated_candidate" and row.get("status") == "validated_candidate"):
                retry_by_id[event_id] = row
            elif existing.get("status") == "validated_candidate" and row.get("status") == "validated_candidate":
                if existing.get("model_response_sha256") != row.get("model_response_sha256"):
                    raise ValueError(f"multiple distinct validated retries for {event_id}; choose one explicitly")
        if set(row_ids) - set(base_by_id):
            raise ValueError(f"retry contains events missing from base: {sorted(set(row_ids) - set(base_by_id))}")
    output_rows, replaced = [], []
    for event_id, base in base_by_id.items():
        event = event_by_id.get(event_id)
        if event is None or base.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source-record hash mismatch for {event_id}")
        retry = retry_by_id.get(event_id)
        if retry is None:
            output_rows.append(base)
            continue
        if base.get("status") != "model_error":
            raise ValueError(f"refusing to replace non-error base row for {event_id}")
        if retry.get("status") != "validated_candidate":
            output_rows.append(base)
            continue
        if retry.get("source_record_sha256") != base.get("source_record_sha256"):
            raise ValueError(f"retry source hash differs for {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        expected_transcript_hash = _sha256(transcript.encode("utf-8"))
        if retry.get("target_transcript_sha256") != expected_transcript_hash:
            raise ValueError(f"retry transcript hash differs for {event_id}")
        turns_by_id = {turn["turn_id"]: turn["text"] for turn in turns}
        retry = validate_layered_outcome(retry, turns_by_id)
        output_rows.append(retry)
        replaced.append(event_id)
    unknown = set(retry_by_id) - set(base_by_id)
    if unknown:
        raise ValueError(f"retry contains events missing from base: {sorted(unknown)}")
    unresolved = [row.get("event_id") for row in output_rows if row.get("status") != "validated_candidate"]
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite layered merge: {output_path}")
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-transcript-outcome-retry-merge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events_sha256": _sha256(event_bytes),
        "base_artifact_sha256": _sha256(base_bytes),
        "base_manifest_sha256": _sha256(base_manifest_bytes),
        "retry_artifact_sha256s": retry_artifact_hashes,
        "retry_manifest_sha256s": retry_manifest_hashes,
        "event_count": len(output_rows),
        "replaced_error_count": len(replaced),
        "replaced_event_ids": replaced,
        "unresolved_error_count": len(unresolved),
        "unresolved_event_ids": unresolved,
        "mention_count": sum(len(row.get("mentions", [])) for row in output_rows),
        "idea_count": sum(len(row.get("ideas", [])) for row in output_rows),
        "link_count": sum(len(row.get("links", [])) for row in output_rows),
        "exact_quote_count": sum(row.get("validation", {}).get("evidence_quote_count", 0) for row in output_rows),
        "outputs_local_only": True,
        "output_sha256": _sha256(output_bytes),
        "limitations": [
            "Retry outputs are assistant-generated transcript interpretations, not semantic gold.",
            "Successful exact quote checks validate substrings only; they do not establish semantic support or completeness.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "replaced_errors": len(replaced), "unresolved_errors": len(unresolved)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--retry", type=Path, action="append", default=None, help="retry artifact; repeat to combine separate event retries")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(merge(
        events_path=args.events,
        base_path=args.base,
        retry_paths=args.retry or [RETRY_DEFAULT],
        output_path=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
