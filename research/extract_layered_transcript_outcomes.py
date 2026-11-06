"""Extract transcript mentions and higher-level forecast ideas as separate layers.

This is an AI-authored transcript-only pilot, not semantic gold. Every mention
is linked to one or more abstract ideas; only the idea layer is a forecast target.
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

from speechprefix.headline_idea_matching import (
    ISSUE_PHRASE_EXTRACTOR_VERSION,
    extract_transcript_ideas,
    issue_phrase_lexicon_sha256,
)
from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/2019-layered-transcript-outcomes-v1.jsonl"
PROVIDER = "openai-codex"
MODEL = "gpt-6-luna"
EXTRACTOR_VERSION = "layered-transcript-outcomes-v1"
_PROMPT_VERSION = "transcript-mentions-to-forecast-ideas-v1"
_SYSTEM_PROMPT = (
    "You are a careful transcript-only research extractor. Return one JSON object and no markdown. "
    "Use only the supplied target-speaker transcript; do not use tools, news, event identity, or outside knowledge."
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
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


def _blind_turns(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    blind, blind_to_source = [], {}
    for index, turn in enumerate(turns):
        blind_id = f"{case_id}:turn:{index:04d}"
        blind.append({"turn_id": blind_id, "text": turn["text"]})
        blind_to_source[blind_id] = turn["turn_id"]
    return blind, blind_to_source


def _prompt(turns: list[dict[str, str]]) -> str:
    instructions = """Read every target-speaker turn. Produce TWO linked layers; do not flatten them into one list.

Layer 1 — `mentions`: transcript-local substantive content units. Capture distinct expressed claims, positions, proposals, arguments, appeals, and meaningful event-specific subjects or supporting details. Keep relevant names/actions/examples when they carry meaning. Do not make separate units for routine greetings, applause, filler, or repeated wording. Each mention has a short faithful paraphrase, a role, salience, and exact evidence.

Layer 2 — `ideas`: higher-level concepts/claims that could serve as retrospective idea-forecast targets. Group related mentions under one idea when they express/support the same concept. Do NOT promote every named person, example, statistic, anecdote, or ceremonial detail into a separate forecast idea. Conversely, do not merge genuinely distinct positions or arguments. An event-specific or ceremonial subject may be retained as an idea with `forecastable=false` if it is independently meaningful; forecastability is a separate, cautious judgment, not a test for whether something belongs in the transcript inventory.

`links` connect each mention to at least one idea: `expresses` for direct expression; `supports`, `illustrates`, `elaborates`, `qualifies`, or `contextualizes` for its relation to the broader idea. One idea may have many mentions; one mention may connect to more than one idea only when warranted. Never create an idea without a supporting linked mention or a mention without a linked idea. Preserve exact quote provenance on mentions; ideas inherit evidence through links.

Mention roles: `core_expression`, `supporting_detail`, `illustrative_example`, `event_specific_detail`, `ceremonial_or_personal`, `other`. Salience: `central`, `secondary`, `brief`. Idea kinds: `policy`, `political`, `factual_claim`, `campaign_appeal`, `ceremonial`, `personal_anecdote`, `other`. Stances: `support`, `oppose`, `describe`, `unclear`. Idea prominence: `central`, `repeated`, `secondary`, `brief`.

Every mention must cite 1–4 exact contiguous quotes copied from supplied turns, using their blind IDs. Exact quote equality will be checked mechanically. Aim for complete coverage of substantive speech content, not exhaustive sentence-by-sentence paraphrase. Use no more than 50 mentions, 20 ideas, and 100 links; do not add content merely to fill a quota. If no substantive content exists, empty arrays are allowed. No self-reported confidence field.

Return exactly this JSON shape:
{
  "mentions": [
    {
      "mention_id": "lowercase_snake_case_id",
      "text": "concise transcript-local paraphrase of one substantive content unit",
      "role": "core_expression" | "supporting_detail" | "illustrative_example" | "event_specific_detail" | "ceremonial_or_personal" | "other",
      "salience": "central" | "secondary" | "brief",
      "evidence": [{"turn_id": "blind ID", "exact_quote": "verbatim contiguous quote"}]
    }
  ],
  "ideas": [
    {
      "idea_id": "lowercase_snake_case_id",
      "concept": "short higher-level concept label",
      "summary": "one concise neutral sentence describing the expressed idea",
      "kind": "policy" | "political" | "factual_claim" | "campaign_appeal" | "ceremonial" | "personal_anecdote" | "other",
      "stance": "support" | "oppose" | "describe" | "unclear",
      "prominence": "central" | "repeated" | "secondary" | "brief",
      "forecastable": true | false
    }
  ],
  "links": [
    {"mention_id": "mention ID", "idea_id": "idea ID", "relation": "expresses" | "supports" | "illustrates" | "elaborates" | "qualifies" | "contextualizes"}
  ],
  "coverage_note": "brief note on any transcript limitation or major ambiguity; otherwise empty string"
}

TRANSCRIPT TURNS:
"""
    return instructions + json.dumps({"turns": turns}, ensure_ascii=False)


def _parse_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def _call_pi(prompt: str, *, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".prompt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        result = subprocess.run([
            executable, "--provider", PROVIDER, "--model", model, "--print", "--mode", "text",
            "--no-tools", "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files",
            "--no-session", "--system-prompt", _SYSTEM_PROMPT, f"@{prompt_path}",
        ], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def _map_layered_result(
    value: Any,
    *,
    blind_turns: list[dict[str, str]],
    blind_to_source: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    for field in ("mentions", "ideas", "links"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"{field} must be a list")
    coverage_note = value.get("coverage_note")
    if not isinstance(coverage_note, str):
        raise ValueError("coverage_note must be a string")
    available_blind = {turn["turn_id"] for turn in blind_turns}
    mentions = []
    for mention in value["mentions"]:
        if not isinstance(mention, dict) or not isinstance(mention.get("evidence"), list):
            raise ValueError("each mention must be an object with evidence")
        evidence = []
        for citation in mention["evidence"]:
            if not isinstance(citation, dict) or citation.get("turn_id") not in available_blind:
                raise ValueError("mention cites an unknown blind turn")
            evidence.append({"turn_id": blind_to_source[citation["turn_id"]], "exact_quote": citation.get("exact_quote")})
        mentions.append({**{key: item for key, item in mention.items() if key != "evidence"}, "evidence": evidence})
    record = {
        "schema_version": 1,
        "artifact_type": "layered-transcript-outcome",
        **metadata,
        "mentions": mentions,
        "ideas": value["ideas"],
        "links": value["links"],
        "coverage_note": coverage_note.strip(),
    }
    turns_by_id = {blind_to_source[turn["turn_id"]]: turn["text"] for turn in blind_turns}
    return validate_layered_outcome(record, turns_by_id)


def run(
    *, events_path: Path, output_path: Path, target_year: str = "2019", timeout: int = 240,
    event_ids: set[str] | None = None, include_weak_labeled: bool = False, model: str = MODEL,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    indexed = {}
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in indexed:
            raise ValueError("event corpus must have unique nonempty event IDs")
        indexed[event_id] = event
    selected = []
    for event_id, event in indexed.items():
        if not str(event.get("event_date", "")).startswith(target_year):
            continue
        if event_ids is not None and event_id not in event_ids:
            continue
        turns = _target_turns(event)
        weak_count = len(extract_transcript_ideas(turns))
        if event_ids is None and not include_weak_labeled and weak_count:
            continue
        selected.append((event_id, event, turns, weak_count))
    if event_ids is not None:
        missing = event_ids - set(indexed)
        if missing:
            raise ValueError(f"requested event IDs are missing: {sorted(missing)}")
    if not selected:
        raise ValueError("no events selected")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite layered outcomes: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    prompt_hash = _sha256((_SYSTEM_PROMPT + "\n" + _prompt([])).encode("utf-8"))
    results = []
    for index, (event_id, event, turns, weak_count) in enumerate(selected, 1):
        blind_id = f"layered-case-{index:04d}"
        blind_turns, blind_to_source = _blind_turns(turns, blind_id)
        prompt = _prompt(blind_turns)
        errors, validated, response_hash = [], None, None
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
                metadata = {
                    "event_id": event_id,
                    "event_date": event.get("event_date"),
                    "source_record_sha256": _canonical_hash(event),
                    "target_transcript_sha256": _sha256(transcript.encode("utf-8")),
                    "target_speaker_turn_count": len(turns),
                    "target_speaker_word_count": sum(len(turn["text"].split()) for turn in turns),
                    "weak_extractor": {
                        "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
                        "lexicon_sha256": issue_phrase_lexicon_sha256(),
                        "idea_count": weak_count,
                    },
                    "extractor": {
                        "version": EXTRACTOR_VERSION,
                        "provider": PROVIDER,
                        "model": model,
                        "metadata_hidden": True,
                        "news_and_weak_labels_hidden": True,
                        "transcript_used_only_as_outcome": True,
                    },
                }
                validated = _map_layered_result(
                    _parse_object(response), blind_turns=blind_turns,
                    blind_to_source=blind_to_source, metadata=metadata,
                )
                validated["model_response_sha256"] = response_hash
                validated["model_attempts"] = attempt + 1
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
                if attempt < 2:
                    prompt += (
                        "\n\nFORMAT REPAIR: The previous response failed validation with: " + str(exc)
                        + ". Repair the previous response below; do not start over or repeat the invalid fields. Use only the allowed role/kind/stance/relation values, preserve valid content, "
                        "and copy every evidence quote as an exact substring of its cited transcript turn. Return the required JSON only.\n"
                        "PREVIOUS RESPONSE TO REPAIR:\n" + response
                    )
        if validated is None:
            output = {
                "schema_version": 1, "artifact_type": "layered-transcript-outcome-error",
                "event_id": event_id, "event_date": event.get("event_date"),
                "source_record_sha256": _canonical_hash(event), "status": "model_error", "errors": errors,
            }
        else:
            validated["status"] = "validated_candidate"
            output = validated
        results.append(output)
        print(f"[{index}/{len(selected)}] {blind_id}: {output['status']} (mentions={len(output.get('mentions', []))}, ideas={len(output.get('ideas', []))})", flush=True)
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-transcript-outcome-extraction-run",
        "created_at": started,
        "extractor_version": EXTRACTOR_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "events_sha256": _sha256(event_bytes),
        "target_year": target_year,
        "selection_policy": "target-year records with zero compact lexical labels unless --include-weak-labeled or --event-id is specified",
        "provider": PROVIDER,
        "model": model,
        "event_count": len(results),
        "validated_event_count": sum(row.get("status") == "validated_candidate" for row in results),
        "model_error_count": sum(row.get("status") == "model_error" for row in results),
        "mention_count": sum(len(row.get("mentions", [])) for row in results),
        "idea_count": sum(len(row.get("ideas", [])) for row in results),
        "link_count": sum(len(row.get("links", [])) for row in results),
        "quote_count": sum(row.get("validation", {}).get("evidence_quote_count", 0) for row in results),
        "forecastable_idea_count": sum(idea.get("forecastable") is True for row in results for idea in row.get("ideas", [])),
        "outputs_local_only": True,
        "limitations": [
            "Transcript mentions and idea nodes are AI-authored candidates, not semantic gold or human-reviewed labels.",
            "Exact quote checks establish substring provenance, not semantic entailment.",
            "Mention coverage and idea abstraction remain unvalidated; a structural contract cannot certify completeness.",
            "The pilot uses the 2019 events left empty by a compact lexical extractor and is not a uniform training/evaluation corpus.",
            "The transcript may make event identity inferable despite hiding event metadata.",
        ],
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "events": len(results), "mentions": manifest["mention_count"], "ideas": manifest["idea_count"], "errors": manifest["model_error_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--target-year", default="2019")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--include-weak-labeled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events, output_path=args.output, target_year=args.target_year,
        timeout=args.timeout, event_ids=set(args.event_ids) if args.event_ids else None,
        include_weak_labeled=args.include_weak_labeled, model=args.model,
    ), indent=2))


if __name__ == "__main__":
    main()
