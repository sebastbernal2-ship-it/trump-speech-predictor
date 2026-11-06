"""Independently identify direct-expression anchors for frozen transcript ideas.

This pass labels only mention→idea `expresses` anchors. Fine-grained relation
labels are deliberately deferred to `extract_layered_relation_edges.py`.
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

from speechprefix.idea_layer.semantic_anchors import build_expression_anchor_record
from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
ENGINE_VERSION = "layered-expression-anchor-pass-v1"
PROMPT_VERSION = "frozen-ideas-direct-expression-anchors-v1"
PROVIDER = "openai-codex"
MODEL = "gpt-6.1-sol"
OUTPUT_DEFAULT = ROOT / "data/source-archives/layered-expression-anchors-v1.jsonl"
_SYSTEM_PROMPT = (
    "You are a careful transcript-only semantic anchor annotator. Use only supplied transcript text and frozen nodes. "
    "Do not use tools, outside knowledge, news, event identity, or forecastability labels. Return one JSON object and no markdown."
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
        blind_id = f"anchor-case-{case_id}:turn:{index:04d}"
        blind.append({"turn_id": blind_id, "text": turn["text"]})
        blind_to_source[blind_id] = turn["turn_id"]
    return blind, blind_to_source


def _prompt(turns: list[dict[str, str]], outcome: dict[str, Any], source_to_blind: dict[str, str]) -> str:
    mentions = []
    for mention in outcome["mentions"]:
        mentions.append({
            **{key: value for key, value in mention.items() if key != "evidence"},
            "evidence": [
                {"turn_id": source_to_blind[item["turn_id"]], "exact_quote": item["exact_quote"]}
                for item in mention["evidence"]
            ],
        })
    ideas = [
        {key: value for key, value in idea.items() if key != "forecastable"}
        for idea in outcome["ideas"]
    ]
    return f"""Identify direct-expression anchors for a frozen mention/idea inventory.

A direct-expression anchor means the supplied transcript mention itself states or directly expresses the supplied idea. Do not include a mention merely because it supports, illustrates, elaborates, qualifies, or contextualizes the idea; those finer relations belong to a separate pass. Do not change, add, remove, or rewrite any mention or idea. The previous relation labels are intentionally not supplied.

For each idea, identify every clear direct-expression mention. If no mention clearly expresses an idea, put that idea ID in `unanchored_idea_ids` instead of inventing an anchor. Every supplied idea ID must occur either in at least one anchor or in `unanchored_idea_ids`, but never both. A mention may directly express more than one idea only when each is independently stated. Do not use idea forecastability to decide anchoring. Exact mention evidence is already included and resolves to the supplied transcript turns; do not create quote text.

Return only {{"anchors": [{{"mention_id": "supplied mention ID", "idea_id": "supplied idea ID"}}], "unanchored_idea_ids": ["supplied idea ID"]}}. Return no other keys.

INPUT:
{json.dumps({"turns": turns, "mentions": mentions, "ideas": ideas}, ensure_ascii=False)}"""


def _parse_result(text: str) -> tuple[list[dict[str, Any]], list[str]]:
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
    if not isinstance(value, dict) or set(value) != {"anchors", "unanchored_idea_ids"}:
        raise ValueError("response must contain exactly anchors and unanchored_idea_ids")
    if not isinstance(value["anchors"], list) or not isinstance(value["unanchored_idea_ids"], list):
        raise ValueError("anchors and unanchored_idea_ids must be lists")
    return value["anchors"], value["unanchored_idea_ids"]


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".anchor-prompt", delete=False) as handle:
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
                raise RuntimeError("provider rejected the anchor request for authentication or account balance")
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty anchor response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def run(
    *,
    events_path: Path,
    outcomes_path: Path,
    output_path: Path,
    event_ids: set[str],
    provider: str = PROVIDER,
    model: str = MODEL,
    timeout: int = 300,
) -> dict[str, Any]:
    if not event_ids:
        raise ValueError("specify one or more --event-id values; anchor extraction is opt-in and bounded")
    events, event_bytes = _read_jsonl(events_path)
    event_by_id = {event.get("event_id"): event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("event source contains duplicate event IDs")
    outcomes, outcomes_bytes = _read_jsonl(outcomes_path)
    outcome_by_id = {row.get("event_id"): row for row in outcomes}
    if len(outcome_by_id) != len(outcomes):
        raise ValueError("outcome artifact contains duplicate event IDs")
    unknown = event_ids - set(outcome_by_id)
    if unknown:
        raise ValueError(f"requested events have no source outcomes: {sorted(unknown)}")
    missing_events = event_ids - set(event_by_id)
    if missing_events:
        raise ValueError(f"requested events are missing from the transcript corpus: {sorted(missing_events)}")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite expression anchor artifact: {output_path}")

    rows, errors = [], []
    for index, event_id in enumerate(sorted(event_ids), 1):
        outcome, event = outcome_by_id[event_id], event_by_id[event_id]
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
        blind, blind_to_source = _blind_turns(turns, f"{index:04d}")
        source_to_blind = {source_id: blind_id for blind_id, source_id in blind_to_source.items()}
        prompt = _prompt(blind, outcome, source_to_blind)
        result_record = None
        attempt_errors = []
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                proposed_anchors, unanchored = _parse_result(response)
                result_record = build_expression_anchor_record(
                    outcome,
                    proposed_anchors,
                    unanchored,
                    provider=provider,
                    model=model,
                    engine_version=ENGINE_VERSION,
                    prompt_version=PROMPT_VERSION,
                    turns_by_id=turns_by_id,
                    response_sha256=response_hash,
                )
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, RuntimeError) and "provider rejected" in str(exc):
                    raise
                result_record = None
                if attempt < 2:
                    prompt += (
                        "\n\nFORMAT REPAIR: Return exactly one JSON object with anchors and unanchored_idea_ids. "
                        "Every supplied idea ID must be covered by one or the other; do not invent direct-expression anchors.\nPREVIOUS RESPONSE:\n" + response
                    )
        if result_record is None:
            errors.append({"event_id": event_id, "errors": attempt_errors})
            print(f"[{index}/{len(event_ids)}] anchor-case-{index:04d}: model_error", flush=True)
            continue
        rows.append(result_record)
        print(f"[{index}/{len(event_ids)}] anchor-case-{index:04d}: anchors={len(result_record['anchors'])}, unanchored={len(result_record['unanchored_idea_ids'])}", flush=True)

    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-expression-anchor-extraction-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "events_sha256": _sha256(event_bytes),
        "source_outcome_artifact_sha256": _sha256(outcomes_bytes),
        "provider": provider,
        "model": model,
        "requested_event_ids": sorted(event_ids),
        "successful_event_count": len(rows),
        "model_error_count": len(errors),
        "anchor_count": sum(len(row["anchors"]) for row in rows),
        "unanchored_idea_count": sum(len(row["unanchored_idea_ids"]) for row in rows),
        "errors": errors,
        "semantic_anchor_quality_validated": False,
        "anchor_completeness_validated": False,
        "outputs_local_only": True,
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "events": len(rows), "model_errors": len(errors), "anchors": sum(len(row["anchors"]) for row in rows), "unanchored_ideas": sum(len(row["unanchored_idea_ids"]) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--event-id", action="append", required=True)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        outcomes_path=args.outcomes,
        output_path=args.output,
        event_ids=set(args.event_id),
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
