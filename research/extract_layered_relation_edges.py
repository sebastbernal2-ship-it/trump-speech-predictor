"""Generate optional fine-grained edges over frozen layered transcript outcomes.

The source outcome's `expresses` links are copied as immutable anchors. This
second model pass sees those anchors but cannot create or modify them; it emits
only optional non-`expresses` mention-to-idea relation candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from speechprefix.idea_layer.relation_edges import (
    FINE_RELATION_DEFINITIONS,
    build_relation_candidate_record,
    direct_expression_anchors,
    validate_relation_candidate_record,
)
from speechprefix.idea_layer.semantic_anchors import validate_expression_anchor_record
from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
ENGINE_VERSION = "layered-fine-relation-edge-pass-v1"
PROMPT_VERSION = "frozen-expresses-anchors-fine-relations-v1"
PROVIDER = "openai-codex"
MODEL = "gpt-6-luna"
OUTPUT_DEFAULT = ROOT / "data/source-archives/layered-fine-relation-edge-candidates-v2.jsonl"
_SYSTEM_PROMPT = (
    "You are a conservative transcript-only relation annotator. Use only the supplied transcript and frozen nodes/anchors. "
    "Do not use tools, news, event identity, outside knowledge, or forecast labels. Return one JSON object and no markdown."
)
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows, raw


def _target_turns(event: dict[str, Any]) -> list[dict[str, str]]:
    target = str(event.get("target_speaker", "")).casefold()
    turns = []
    for turn in event.get("turns", []):
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


def _blind_turns(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    blind, blind_to_source = [], {}
    for index, turn in enumerate(turns):
        blind_id = f"{case_id}:turn:{index:04d}"
        blind.append({"turn_id": blind_id, "text": turn["text"]})
        blind_to_source[blind_id] = turn["turn_id"]
    return blind, blind_to_source


def _prompt(
    blind_turns: list[dict[str, str]],
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    blind_anchors: list[dict[str, str]],
) -> str:
    anchored_idea_ids = {item["idea_id"] for item in anchor_record["anchors"]}
    content = {
        "transcript_turns": blind_turns,
        "frozen_mentions": outcome["mentions"],
        "frozen_ideas": [
            {key: value for key, value in idea.items() if key != "forecastable"}
            for idea in outcome["ideas"] if idea["idea_id"] in anchored_idea_ids
        ],
        "direct_expression_anchors": blind_anchors,
        "unanchored_idea_ids": anchor_record["unanchored_idea_ids"],
    }
    return f"""Annotate optional fine-grained relations over an already frozen mention/idea inventory.

The direct-expression anchors are fixed input from a separate anchor pass. Do not add, remove, relabel, or reinterpret anchors; do not return `expresses`. Ideas listed as unanchored have no direct-expression anchor in that pass and are not eligible for fine edges. Your task is only to propose additional relations from a mention to an anchored idea.

Relation definitions:
{json.dumps(FINE_RELATION_DEFINITIONS, ensure_ascii=False, indent=2)}

Guidance:
- Return only clear, useful relations. A mention being in the same speech, topic, paragraph, or event is not enough.
- Prefer no edge over an ambiguous edge. An omitted edge means "not proposed here", not "unrelated".
- For one mention/idea pair, choose at most one best relation; do not emit a second label for the same pair.
- A direct-expression anchor pair is already represented and must not be repeated.
- Do not create or modify mentions, ideas, evidence, nodes, or IDs. Select only supplied mention/idea IDs.
- Relation proposals are not semantic gold and have no confidence score.

Return exactly {{"edges": [{{"mention_id": "supplied ID", "idea_id": "supplied ID", "relation": "supports|illustrates|elaborates|qualifies|contextualizes"}}]}}. Return an empty list if no additional relation is sufficiently clear. Maximum 1,200 edges.

INPUT:
{json.dumps(content, ensure_ascii=False)}"""


def _parse_edges(text: str) -> list[dict[str, str]]:
    decoder = json.JSONDecoder()
    value = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if not isinstance(value, dict) or set(value) != {"edges"} or not isinstance(value["edges"], list):
        raise ValueError("response must be a JSON object containing only an edges list")
    edges = value["edges"]
    if len(edges) > 1200:
        raise ValueError("response exceeds 1,200 relation edges")
    return edges


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".relation-prompt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        result = subprocess.run([
            executable, "--provider", provider, "--model", model, "--print", "--mode", "text",
            "--no-tools", "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files",
            "--no-session", "--system-prompt", _SYSTEM_PROMPT, f"@{prompt_path}",
        ], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            output = (result.stdout + "\n" + result.stderr).casefold()
            if any(marker in output for marker in ("insufficient balance", "invalid api key", "unauthorized", "authentication failed", "forbidden")):
                raise RuntimeError("provider rejected the relation request for authentication or account balance")
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty relation response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def run(
    *,
    events_path: Path,
    outcomes_path: Path,
    anchors_path: Path,
    output_path: Path,
    event_ids: set[str],
    provider: str = PROVIDER,
    model: str = MODEL,
    timeout: int = 300,
) -> dict[str, Any]:
    if not event_ids:
        raise ValueError("specify one or more --event-id values; relation extraction is opt-in and bounded")
    events, event_bytes = _read_jsonl(events_path)
    event_by_id = {event.get("event_id"): event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("event source contains duplicate event IDs")
    outcomes, outcomes_bytes = _read_jsonl(outcomes_path)
    outcome_by_id = {row.get("event_id"): row for row in outcomes}
    if len(outcome_by_id) != len(outcomes):
        raise ValueError("outcome artifact contains duplicate event IDs")
    anchor_rows, anchors_bytes = _read_jsonl(anchors_path)
    anchor_by_id = {row.get("event_id"): row for row in anchor_rows}
    if len(anchor_by_id) != len(anchor_rows):
        raise ValueError("expression-anchor artifact contains duplicate event IDs")
    unknown = event_ids - set(outcome_by_id)
    if unknown:
        raise ValueError(f"requested events have no source outcomes: {sorted(unknown)}")
    missing_events = event_ids - set(event_by_id)
    if missing_events:
        raise ValueError(f"requested events are missing from the transcript corpus: {sorted(missing_events)}")
    missing_anchors = event_ids - set(anchor_by_id)
    if missing_anchors:
        raise ValueError(f"requested events have no expression-anchor artifact: {sorted(missing_anchors)}")

    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite relation candidate artifact: {output_path}")

    rows, errors = [], []
    for index, event_id in enumerate(sorted(event_ids), 1):
        outcome, anchor_record, event = outcome_by_id[event_id], anchor_by_id[event_id], event_by_id[event_id]
        if outcome.get("status") != "validated_candidate":
            raise ValueError(f"source outcome is not structurally validated: {event_id}")
        if outcome.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source event hash mismatch: {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        if outcome.get("target_transcript_sha256") != _sha256(transcript.encode("utf-8")):
            raise ValueError(f"target transcript hash mismatch: {event_id}")
        turns_by_id = {turn["turn_id"]: turn["text"] for turn in turns}
        validate_layered_outcome(outcome, turns_by_id)
        validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
        source_anchors = direct_expression_anchors(anchor_record)
        blind_turns, blind_to_source = _blind_turns(turns, f"relation-case-{index:04d}")
        source_to_blind = {source_id: blind_id for blind_id, source_id in blind_to_source.items()}
        blind_mentions = []
        for mention in outcome["mentions"]:
            blind_mentions.append({
                **{key: value for key, value in mention.items() if key != "evidence"},
                "evidence": [
                    {"turn_id": source_to_blind[citation["turn_id"]], "exact_quote": citation["exact_quote"]}
                    for citation in mention["evidence"]
                ],
            })
        blinded_outcome = {"mentions": blind_mentions, "ideas": outcome["ideas"]}
        blind_anchors = [
            {"mention_id": anchor["mention_id"], "idea_id": anchor["idea_id"]}
            for anchor in source_anchors
        ]
        prompt = _prompt(blind_turns, blinded_outcome, anchor_record, blind_anchors)
        relation_record = None
        attempt_errors = []
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                proposed_edges = _parse_edges(response)
                relation_record = build_relation_candidate_record(
                    outcome,
                    anchor_record,
                    proposed_edges,
                    provider=provider,
                    model=model,
                    engine_version=ENGINE_VERSION,
                    prompt_version=PROMPT_VERSION,
                    turns_by_id=turns_by_id,
                )
                relation_record["extractor"]["response_sha256"] = response_hash
                validate_relation_candidate_record(relation_record, outcome, anchor_record, turns_by_id)
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, RuntimeError) and "provider rejected" in str(exc):
                    raise
                relation_record = None
                if attempt < 2:
                    prompt += (
                        "\n\nFORMAT REPAIR: Return exactly one JSON object with an edges list. Use only supplied IDs, "
                        "one non-expresses relation per non-anchor pair, or omit the pair. No outside sources.\nPREVIOUS RESPONSE:\n" + response
                    )
        if relation_record is None:
            errors.append({"event_id": event_id, "errors": attempt_errors})
            print(f"[{index}/{len(event_ids)}] relation-case-{index:04d}: model_error", flush=True)
            continue
        rows.append(relation_record)
        print(f"[{index}/{len(event_ids)}] relation-case-{index:04d}: edges={len(relation_record['edges'])}", flush=True)

    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-fine-relation-edge-extraction-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "events_sha256": _sha256(event_bytes),
        "source_outcome_artifact_sha256": _sha256(outcomes_bytes),
        "source_anchor_artifact_sha256": _sha256(anchors_bytes),
        "provider": provider,
        "model": model,
        "requested_event_ids": sorted(event_ids),
        "successful_event_count": len(rows),
        "model_error_count": len(errors),
        "candidate_edge_count": sum(len(row["edges"]) for row in rows),
        "relation_counts": {
            relation: sum(edge["relation"] == relation for row in rows for edge in row["edges"])
            for relation in sorted(FINE_RELATION_DEFINITIONS)
        },
        "errors": errors,
        "semantic_relations_validated": False,
        "outputs_local_only": True,
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "events": len(rows),
        "model_errors": len(errors),
        "candidate_edges": sum(len(row["edges"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--event-id", action="append", required=True)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        outcomes_path=args.outcomes,
        anchors_path=args.anchors,
        output_path=args.output,
        event_ids=set(args.event_id),
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
