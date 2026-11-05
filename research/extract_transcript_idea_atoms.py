"""Run a transcript-only, open-vocabulary idea-atom extraction pilot via pi.

This is an AI-authored diagnostic label source, not semantic ground truth. By
 default it processes only 2019 events with zero labels from the compact lexical
extractor. Event metadata is hidden from the model; evidence quotes are checked
against the cited target-speaker turns before output is accepted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
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

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/2019-weak-empty-semantic-idea-atoms-v1.jsonl"
EXTRACTOR_VERSION = "transcript-semantic-idea-atoms-v1"
_SYSTEM_PROMPT = (
    "You are a careful transcript-only research extractor. Return exactly one JSON object and no markdown. "
    "Do not use tools, external information, event metadata, prior knowledge about the event, or news. "
    "The supplied transcript is the complete available target-speaker text for one event."
)
_PROMPT_VERSION = "transcript-only-open-vocabulary-atoms-v1"
_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_STATUSES = {"atoms_found", "no_supported_idea", "insufficient_transcript"}
_KINDS = {"policy", "political", "factual_claim", "campaign_appeal", "ceremonial", "personal_anecdote", "other"}
_STANCES = {"support", "oppose", "describe", "unclear"}
_PROMINENCE = {"central", "repeated", "secondary", "brief"}
_CONFIDENCE = {"high", "medium", "low"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
    return rows


def _target_turns(record: dict[str, Any]) -> list[dict[str, str]]:
    target = str(record.get("target_speaker", "")).casefold()
    turns = []
    for turn in record.get("turns", []):
        if not isinstance(turn, dict):
            continue
        is_target = turn.get("speaker_role") == "target_speaker" or (
            target and str(turn.get("speaker_name", "")).casefold() == target
        )
        text = turn.get("text") or turn.get("raw_text")
        turn_id = turn.get("turn_id")
        if is_target and isinstance(text, str) and text.strip():
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError("target-speaker transcript turn is missing a turn_id")
            turns.append({"turn_id": turn_id, "text": text.strip()})
    return turns


def _blind_turns(turns: list[dict[str, str]], blind_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    mapping = {}
    result = []
    for index, turn in enumerate(turns):
        blind_turn_id = f"{blind_id}:turn:{index:04d}"
        mapping[blind_turn_id] = turn["turn_id"]
        result.append({"turn_id": blind_turn_id, "text": turn["text"]})
    return result, mapping


def _prompt(blind_turns: list[dict[str, str]]) -> str:
    instructions = """Read every supplied target-speaker turn before answering. Extract open-vocabulary, transcript-grounded idea atoms: concise canonical concepts that capture a distinct proposition, policy position, argument, factual/political claim, or appeal actually expressed. Generalize paraphrases to one concept, but do not merge unrelated ideas or invent motives/details. Include substantive event-specific or ceremonial content as an atom when it carries meaning, and mark its kind and forecastability separately. Greetings, applause, filler, and routine thanks alone are not ideas. Do not require a phrase to match a fixed lexicon.

Every atom must cite at least one exact, contiguous verbatim quote from a supplied turn. Use the supplied blind turn IDs. Quotes must be copied exactly, including spelling and punctuation. Do not cite a turn that does not support the atom. Use an empty status only when appropriate: `no_supported_idea` means you reviewed the full supplied transcript and found no substantive atom; `insufficient_transcript` means the text is too incomplete or corrupted to judge. Do not use either status merely because an idea is ceremonial or non-policy.

Return exactly this JSON shape, with no other text:
{
  "status": "atoms_found" | "no_supported_idea" | "insufficient_transcript",
  "empty_reason": "brief reason or empty string when atoms are present",
  "atoms": [
    {
      "idea_id": "lowercase_snake_case_id",
      "concept": "short canonical concept name",
      "summary": "one concise neutral sentence describing what the speaker expresses",
      "kind": "policy" | "political" | "factual_claim" | "campaign_appeal" | "ceremonial" | "personal_anecdote" | "other",
      "stance": "support" | "oppose" | "describe" | "unclear",
      "prominence": "central" | "repeated" | "secondary" | "brief",
      "confidence": "high" | "medium" | "low",
      "forecastable": true | false,
      "evidence": [{"turn_id": "blind turn ID", "exact_quote": "verbatim substring"}]
    }
  ]
}

Use `forecastable=false` for purely ceremonial, incidental, or personal material that is not a general idea likely to be forecast from pre-event context; otherwise make a cautious judgment. Keep the atom inventory focused (usually 1–10 atoms, never more than 15). Prominence reflects the transcript, not expected forecastability.

TRANSCRIPT TURNS:
"""
    return instructions + json.dumps(blind_turns, ensure_ascii=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def _validate_response(value: Any, blind_turns: list[dict[str, str]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    status, atoms, empty_reason = value.get("status"), value.get("atoms"), value.get("empty_reason")
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValueError("status is unsupported")
    if not isinstance(atoms, list) or len(atoms) > 15:
        raise ValueError("atoms must be a list of at most 15 items")
    if not isinstance(empty_reason, str):
        raise ValueError("empty_reason must be a string")
    if status == "atoms_found" and not atoms:
        raise ValueError("atoms_found requires at least one atom")
    if status != "atoms_found" and atoms:
        raise ValueError("empty status must not include atoms")
    if status != "atoms_found" and not empty_reason.strip():
        raise ValueError("empty status requires an explanation")

    available_turns = {turn["turn_id"]: turn["text"] for turn in blind_turns}
    idea_ids: set[str] = set()
    checked_atoms = []
    for index, atom in enumerate(atoms):
        field = f"atoms[{index}]"
        if not isinstance(atom, dict):
            raise ValueError(f"{field} must be an object")
        for name in ("idea_id", "concept", "summary"):
            if not isinstance(atom.get(name), str) or not atom[name].strip():
                raise ValueError(f"{field}.{name} must be nonempty text")
        idea_id = atom["idea_id"]
        if not _ID.fullmatch(idea_id) or idea_id in idea_ids:
            raise ValueError(f"{field}.idea_id must be unique lowercase snake_case")
        idea_ids.add(idea_id)
        for name, allowed in (("kind", _KINDS), ("stance", _STANCES), ("prominence", _PROMINENCE), ("confidence", _CONFIDENCE)):
            if not isinstance(atom.get(name), str) or atom[name] not in allowed:
                raise ValueError(f"{field}.{name} is unsupported")
        if not isinstance(atom.get("forecastable"), bool):
            raise ValueError(f"{field}.forecastable must be boolean")
        evidence = atom.get("evidence")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 4:
            raise ValueError(f"{field}.evidence must contain 1–4 quotes")
        checked_evidence = []
        for evidence_index, citation in enumerate(evidence):
            evidence_field = f"{field}.evidence[{evidence_index}]"
            if not isinstance(citation, dict):
                raise ValueError(f"{evidence_field} must be an object")
            turn_id, quote = citation.get("turn_id"), citation.get("exact_quote")
            if not isinstance(turn_id, str) or turn_id not in available_turns:
                raise ValueError(f"{evidence_field} cites an unknown target-speaker turn")
            if not isinstance(quote, str) or not quote.strip() or quote not in available_turns[turn_id]:
                raise ValueError(f"{evidence_field} quote is not an exact transcript substring")
            checked_evidence.append({"turn_id": turn_id, "exact_quote": quote})
        checked_atoms.append({
            "idea_id": idea_id,
            "concept": atom["concept"].strip(),
            "summary": atom["summary"].strip(),
            "kind": atom["kind"],
            "stance": atom["stance"],
            "prominence": atom["prominence"],
            "confidence": atom["confidence"],
            "forecastable": atom["forecastable"],
            "evidence": checked_evidence,
        })
    return {"status": status, "empty_reason": empty_reason.strip(), "atoms": checked_atoms}


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required for this research extraction run")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".prompt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        command = [
            executable, "--provider", provider, "--model", model,
            "--print", "--mode", "text", "--no-tools", "--no-extensions", "--no-mcp",
            "--no-skills", "--no-context-files", "--no-session", "--system-prompt", _SYSTEM_PROMPT,
            f"@{prompt_path}",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def _extract_one(
    event: dict[str, Any], *, blind_id: str, provider: str, model: str, timeout: int,
) -> dict[str, Any]:
    turns = _target_turns(event)
    blind_turns, turn_map = _blind_turns(turns, blind_id)
    prompt = _prompt(blind_turns)
    errors = []
    for attempt in range(2):
        try:
            response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
            validated = _validate_response(_parse_json_object(response), blind_turns)
            return {**validated, "response_sha256": _sha256(response.encode("utf-8")), "attempts": attempt + 1}
        except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            if attempt == 0:
                prompt += "\n\nFORMAT REPAIR: Your previous response failed validation (" + errors[-1] + "). Re-read the transcript above and return a corrected JSON object only."
    return {
        "status": "model_error", "empty_reason": "", "atoms": [], "attempts": 2,
        "error_types": errors, "response_sha256": None,
    }


def run(
    *, events_path: Path, output_path: Path, target_year: str = "2019",
    provider: str = "openai-codex", model: str = "gpt-6-luna", timeout: int = 180,
    event_ids: set[str] | None = None, include_weak_labeled: bool = False,
) -> dict[str, Any]:
    events = _read_jsonl(events_path)
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
        weak_ideas = extract_transcript_ideas(turns)
        if event_ids is None and not include_weak_labeled and weak_ideas:
            continue
        selected.append((event_id, event, turns, len(weak_ideas)))
    if event_ids is not None:
        missing = event_ids - set(indexed)
        if missing:
            raise ValueError(f"requested event IDs are missing: {sorted(missing)}")
    if not selected:
        raise ValueError("no events selected for extraction")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite extraction output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(timezone.utc).isoformat()
    prompt_sha256 = _sha256((_SYSTEM_PROMPT + "\n" + _prompt([])).encode("utf-8"))
    completed = []
    for index, (event_id, event, turns, weak_count) in enumerate(selected, 1):
        blind_id = f"atom-case-{index:04d}"
        result = _extract_one(event, blind_id=blind_id, provider=provider, model=model, timeout=timeout)
        _blind_turns_for_mapping, turn_map = _blind_turns(turns, blind_id)
        mapped_atoms = []
        for atom in result.get("atoms", []):
            mapped_atoms.append({
                **{key: value for key, value in atom.items() if key != "evidence"},
                "evidence": [
                    {"turn_id": turn_map[item["turn_id"]], "exact_quote": item["exact_quote"]}
                    for item in atom["evidence"]
                ],
            })
        transcript_text = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        output_row = {
            "schema_version": 1,
            "artifact_type": "assistant-extracted-transcript-idea-atoms",
            "extractor_version": EXTRACTOR_VERSION,
            "review_status": "assistant-generated-transcript-only-diagnostic",
            "event_id": event_id,
            "event_date": event.get("event_date"),
            "source_record_sha256": _canonical_hash(event),
            "target_transcript_sha256": _sha256(transcript_text.encode("utf-8")),
            "target_speaker_turn_count": len(turns),
            "target_speaker_word_count": sum(len(turn["text"].split()) for turn in turns),
            "weak_extractor": {
                "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
                "lexicon_sha256": issue_phrase_lexicon_sha256(),
                "idea_count": weak_count,
            },
            "reviewer": {
                "reviewer_type": "AI assistant",
                "provider": provider,
                "model": model,
                "reviewed_at": run_started,
                "full_available_target_speaker_transcript_read": True,
                "event_metadata_and_event_id_hidden_during_extraction": True,
                "news_and_weak_labels_hidden_during_extraction": True,
                "forecastable_is_interpretive": True,
            },
            "extraction_status": result["status"],
            "empty_reason": result.get("empty_reason", ""),
            "atoms": mapped_atoms,
            "model_response_sha256": result.get("response_sha256"),
            "model_attempts": result["attempts"],
            "model_error_types": result.get("error_types", []),
        }
        completed.append(output_row)
        print(f"[{index}/{len(selected)}] {blind_id}: {result['status']} ({len(mapped_atoms)} atoms)", flush=True)

    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in completed).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "transcript-semantic-idea-atom-extraction-run",
        "created_at": run_started,
        "extractor_version": EXTRACTOR_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "events_sha256": _sha256(events_path.read_bytes()),
        "selected_event_count": len(selected),
        "target_year": target_year,
        "selection_policy": "target-year records with zero compact lexical extractor labels, unless --include-weak-labeled or --event-id is specified",
        "weak_extractor_version": ISSUE_PHRASE_EXTRACTOR_VERSION,
        "weak_extractor_lexicon_sha256": issue_phrase_lexicon_sha256(),
        "provider": provider,
        "model": model,
        "pi_cli": shutil.which("pi"),
        "status_counts": {
            status: sum(row["extraction_status"] == status for row in completed)
            for status in sorted(_STATUSES | {"model_error"})
        },
        "atom_count": sum(len(row["atoms"]) for row in completed),
        "forecastable_atom_count": sum(atom["forecastable"] for row in completed for atom in row["atoms"]),
        "output_sha256": _sha256(output_bytes),
        "outputs_local_only": True,
        "limitations": [
            "Assistant-generated interpretations are not objective semantic gold or human-reviewed labels.",
            "This pilot selects events with zero weak lexical labels; its coverage is not comparable to applying this extractor uniformly to all events.",
            "Forecastability is a model judgment, not evidence that an idea was predictable from the archived context.",
            "Transcript content can make event identity inferable even though event IDs, dates, titles, forecasts, and news are hidden.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), **manifest["status_counts"], "atom_count": manifest["atom_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--target-year", default="2019")
    parser.add_argument("--provider", default="openai-codex")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--include-weak-labeled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        output_path=args.output,
        target_year=args.target_year,
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
        event_ids=set(args.event_ids) if args.event_ids else None,
        include_weak_labeled=args.include_weak_labeled,
    ), indent=2))


if __name__ == "__main__":
    main()
