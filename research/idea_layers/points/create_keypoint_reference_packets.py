"""Create local, blind transcript packets and blank reference-annotation forms.

No model calls or generated idea labels are used. Packet transcripts and forms
are written under Git-ignored data/source-archives/ and never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/keypoint-reference-sample-v2"
PROTOCOL = ROOT / "docs/idea-layer/points/human-reference-annotation-protocol-v2.md"
SAMPLE = (
    {
        "packet_id": "packet-01",
        "event_id": "dt47-25-72a1d6e3",
        "stratum": "short remarks / response to a public-safety event",
        "expected_words": 733,
        "expected_title_prefix": "Remarks: Donald Trump Addresses the National Guard Shooting in Washington",
    },
    {
        "packet_id": "packet-02",
        "event_id": "dt47-25-e856e24c",
        "stratum": "medium ceremonial remarks",
        "expected_words": 2013,
        "expected_title_prefix": "Remarks: Donald Trump Welcomes the Champion Florida Panthers",
    },
    {
        "packet_id": "packet-03",
        "event_id": "dt47-25-8623939c",
        "stratum": "long emergency briefing",
        "expected_words": 7700,
        "expected_title_prefix": "Remarks: Donald Trump Receives a Briefing on the California Wildfires",
    },
)
CSV_FIELDS = [
    "annotator_point_id",
    "point_summary",
    "kind",
    "speaker_relation",
    "prominence",
    "evidence_turn_ids",
    "exact_quote",
    "broader_point_id",
    "relation_to_broader_point",
    "boundary_note",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _read_source(path: Path) -> tuple[dict[str, tuple[dict[str, Any], bytes]], str]:
    records: dict[str, tuple[dict[str, Any], bytes]] = {}
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at source line {line_number}: {exc}") from exc
            if row.get("event_id") in {item["event_id"] for item in SAMPLE}:
                records[row["event_id"]] = (row, raw_line)
    return records, _sha256(path.read_bytes())


def _existing_idea_artifacts(source_archives: Path) -> set[str]:
    """Collect IDs with existing layered idea inventories; do not inspect labels."""
    found: set[str] = set()
    for path in source_archives.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and isinstance(row.get("ideas"), list):
                        event_id = row.get("event_id")
                        if isinstance(event_id, str):
                            found.add(event_id)
        except (OSError, UnicodeDecodeError):
            continue
    return found


def _prior_model_artifact_hits(event_ids: set[str]) -> list[dict[str, str]]:
    patterns = (
        ROOT / "data/rollcall-weak-speech-plans*.jsonl",
        ROOT / "data/rollcall-speaker-idea-evaluation*.json",
        ROOT / "data/rollcall-speaker-idea-event-evaluation*.json",
        ROOT / "data/rollcall-transcript-idea-audit*.json",
        ROOT / "data/rollcall-assistant-outcomes*.jsonl",
        ROOT / "data/rollcall-assistant-reviewed-outcomes*.jsonl",
        ROOT / "data/rollcall-assistant-plan-drafts*.jsonl",
        ROOT / "data/rollcall-plan-review*.jsonl",
        ROOT / "data/rollcall-theme-only-assistant-pilot*.jsonl",
        ROOT / "data/source-archives/*.jsonl",
    )
    paths = sorted({path for pattern in patterns for path in pattern.parent.glob(pattern.name)})
    hits = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        text = raw.decode("utf-8", errors="ignore")
        matching_ids = sorted(event_id for event_id in event_ids if event_id in text)
        if matching_ids:
            hits.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(raw),
                "matching_event_ids": ",".join(matching_ids),
            })
    return hits


def _build_packet(packet_id: str, row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int, int, int]:
    target = row.get("target_speaker")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{packet_id}: target_speaker is missing")
    source_turns = [
        turn for turn in row.get("turns", [])
        if isinstance(turn.get("text"), str) and turn["text"].strip()
    ]
    if not source_turns:
        raise ValueError(f"{packet_id}: no transcript turns")
    other_speakers: dict[str, str] = {}
    mapping: list[dict[str, Any]] = []
    sections = [
        f"ANONYMOUS TRANSCRIPT PACKET {packet_id}\n",
        "Read the complete packet before annotating. Turn IDs are packet-local. "
        "Annotate substantive points made only by [TARGET]; [CONTEXT_*] turns are provided for context.\n",
    ]
    target_texts = []
    target_count = 0
    context_count = 0
    for index, turn in enumerate(source_turns, 1):
        blind_id = f"T{index:04d}"
        speaker_name = turn.get("speaker_name")
        is_target = speaker_name == target
        if is_target:
            blind_speaker = "TARGET"
            target_texts.append(turn["text"])
            target_count += 1
        else:
            key = str(speaker_name or turn.get("speaker_role") or "unknown")
            if key not in other_speakers:
                other_speakers[key] = f"CONTEXT_{len(other_speakers) + 1:02d}"
            blind_speaker = other_speakers[key]
            context_count += 1
        text = turn["text"]
        sections.extend([f"\n[{blind_id} {blind_speaker}]\n", text, "\n"])
        mapping.append({
            "blind_turn_id": blind_id,
            "speaker_role": "target" if is_target else "context",
            "source_turn_id": turn.get("turn_id"),
            "source_sequence": turn.get("sequence"),
            "text_sha256": _sha256(text.encode("utf-8")),
        })
    full_target_text = "\n".join(target_texts)
    return "".join(sections), mapping, _count_words(full_target_text), target_count, context_count


def create_packets(source_path: Path = SOURCE_DEFAULT, output_dir: Path = OUTPUT_DEFAULT) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen packet directory: {output_dir}")
    records, source_file_hash = _read_source(source_path)
    missing = [item["event_id"] for item in SAMPLE if item["event_id"] not in records]
    if missing:
        raise ValueError(f"sample event IDs missing from source: {missing}")
    prior_ideas = _existing_idea_artifacts(ROOT / "data/source-archives")
    prior_hits = sorted(item["event_id"] for item in SAMPLE if item["event_id"] in prior_ideas)
    if prior_hits:
        raise ValueError(f"sample contains events with existing layered idea inventories: {prior_hits}")
    model_artifact_hits = _prior_model_artifact_hits({item["event_id"] for item in SAMPLE})
    if model_artifact_hits:
        raise ValueError(f"sample event IDs occur in prior model artifacts: {model_artifact_hits}")

    output_dir.mkdir(parents=True)
    (output_dir / "packets").mkdir()
    for annotator in ("annotator-A", "annotator-B"):
        (output_dir / annotator).mkdir()
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    (output_dir / "ANNOTATION-GUIDE.md").write_text(protocol_text, encoding="utf-8")
    freeze = {
        "benchmark_version": "human-reference-conversation-points-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Fixed 2025 sample chosen across target-speaker word-count and event-format strata. "
            "Selected IDs were checked absent from prior layered idea inventories and the model-artifact "
            "scan documented below. The previously exposed Lexington case is excluded."
        ),
        "source_file": str(source_path.resolve()),
        "source_file_sha256": source_file_hash,
        "annotation_protocol": "docs/idea-layer/points/human-reference-annotation-protocol-v2.md",
        "annotation_protocol_sha256": _sha256(protocol_text.encode("utf-8")),
        "prior_idea_inventory_check": {
            "scope": "root data/source-archives/*.jsonl records with an ideas array",
            "sample_event_ids_found": prior_hits,
            "semantic_inspection_of_prior_labels": False,
        },
        "prior_model_artifact_scan": {
            "scope": "weak speech plans, speaker-idea evaluations, transcript audits, assistant outcomes/drafts, plan/theme pilots, and root source-archive JSONL",
            "hits": model_artifact_hits,
            "model_output_contents_inspected": False,
        },
        "annotators": ["annotator-A", "annotator-B"],
        "speeches": [],
    }
    for selection in SAMPLE:
        row, raw_record = records[selection["event_id"]]
        title = row.get("title", "")
        if not title.startswith(selection["expected_title_prefix"]):
            raise ValueError(f"{selection['packet_id']}: unexpected title; sample freeze mismatch")
        text, turn_map, word_count, target_turn_count, context_turn_count = _build_packet(selection["packet_id"], row)
        if word_count != selection["expected_words"]:
            raise ValueError(
                f"{selection['packet_id']}: expected {selection['expected_words']} words, found {word_count}"
            )
        packet_path = output_dir / "packets" / f"{selection['packet_id']}.txt"
        packet_bytes = text.encode("utf-8")
        packet_path.write_bytes(packet_bytes)
        speech = {
            "packet_id": selection["packet_id"],
            "event_id": selection["event_id"],
            "event_date": row.get("event_date"),
            "title": title,
            "source_url": row.get("source_url"),
            "selection_stratum": selection["stratum"],
            "target_speaker": row.get("target_speaker"),
            "packet_turn_count": len(turn_map),
            "target_speaker_turn_count": target_turn_count,
            "context_turn_count": context_turn_count,
            "target_speaker_word_count": word_count,
            "source_record_sha256": _sha256(raw_record),
            "target_transcript_sha256": _sha256("\n".join(
                turn["text"] for turn in row["turns"] if turn.get("speaker_name") == row.get("target_speaker")
            ).encode("utf-8")),
            "packet_path": str(packet_path.relative_to(output_dir)),
            "packet_sha256": _sha256(packet_bytes),
            "turn_mapping": turn_map,
        }
        freeze["speeches"].append(speech)
        for annotator in ("annotator-A", "annotator-B"):
            form_path = output_dir / annotator / f"{selection['packet_id']}-points.csv"
            with form_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
            speech.setdefault("blank_forms", {})[annotator] = {
                "path": str(form_path.relative_to(output_dir)),
                "sha256": _sha256(form_path.read_bytes()),
            }
    freeze_path = output_dir / "FREEZE-MANIFEST.json"
    freeze_bytes = (json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    freeze_path.write_bytes(freeze_bytes)
    return {"manifest": str(freeze_path), "manifest_sha256": _sha256(freeze_bytes), "speeches": freeze["speeches"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    try:
        result = create_packets(args.source, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"packet creation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "manifest": result["manifest"],
        "manifest_sha256": result["manifest_sha256"],
        "speeches": [
            {key: speech[key] for key in (
                "packet_id", "event_id", "event_date", "selection_stratum",
                "target_speaker_turn_count", "target_speaker_word_count", "packet_sha256",
            )}
            for speech in result["speeches"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
