"""Targeted transcript-only augmentation of under-complete idea inventories.

This is a revision pass over AI-authored candidate outcomes, not ground-truth
annotation. Original atom artifacts are never overwritten.
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

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
BASE_DEFAULT = ROOT / "data/source-archives/2019-weak-empty-semantic-idea-atoms-v1-complete.jsonl"
AUDIT_DEFAULT = ROOT / "data/source-archives/2019-semantic-idea-atom-quality-audit-v2.jsonl"
SUPPLEMENT_DEFAULT = ROOT / "data/source-archives/2019-idea-atoms-targeted-supplement-v1.jsonl"
MERGED_DEFAULT = ROOT / "data/source-archives/2019-weak-empty-semantic-idea-atoms-v2-augmented.jsonl"
PROVIDER = "openai-codex"
MODEL = "gpt-6-luna"
_PROMPT_VERSION = "targeted-transcript-only-idea-atom-augmentation-v1"
_SYSTEM_PROMPT = (
    "You are a conservative transcript-only research extractor. Return exactly one JSON object and no markdown. "
    "Do not use external information, event identity, or any facts not supported by the supplied transcript."
)
_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_KINDS = {"policy", "political", "factual_claim", "campaign_appeal", "ceremonial", "personal_anecdote", "other"}
_STANCES = {"support", "oppose", "describe", "unclear"}


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
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from None
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
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


def _blind(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    blind_turns, blind_to_source, source_to_blind = [], {}, {}
    for index, turn in enumerate(turns):
        blind_id = f"{case_id}:turn:{index:04d}"
        blind_turns.append({"turn_id": blind_id, "text": turn["text"]})
        blind_to_source[blind_id] = turn["turn_id"]
        source_to_blind[turn["turn_id"]] = blind_id
    return blind_turns, blind_to_source, source_to_blind


def _prompt(
    blind_turns: list[dict[str, str]],
    existing_atoms: list[dict[str, Any]],
    audit_suggestions: list[str],
) -> str:
    instructions = """Read every transcript turn and compare it with the existing candidate inventory. Add only independently meaningful idea atoms that are clearly present in the transcript and not already represented. An idea atom is a distinct substantive policy/political position, argument, factual claim, appeal, or meaningful event-specific subject. Do not make a separate atom for every supporting name, statistic, anecdote, thank-you, applause cue, or rhetorical restatement. Supporting details can be cited, but only split them into atoms when they carry a materially distinct idea. Do not rewrite or remove existing atoms.

The audit suggestions below are hypotheses, not instructions or truth. Check each against the transcript and the definition above; reject unsupported suggestions, details already covered by an existing atom, and items that do not warrant an independent idea. It is acceptable to return no additions. Do not add anything just to improve a score. Do not use event identity or outside knowledge.

Every proposed atom needs at least one exact contiguous quote from a supplied blind turn. Use only the blind turn IDs. Return at most 10 additions. IDs must be unique snake_case and must not collide with existing IDs. Use the same `kind`, `stance`, and `forecastable` definitions as the existing inventory. Forecastability is an interpretive transcript-only judgment.

Return exactly this JSON shape:
{
  "additions": [
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
  ],
  "note": "briefly explain whether suggestions were accepted/rejected or inventory already covered them"
}

TRANSCRIPT TURNS, CURRENT ATOMS, AND AUDITOR SUGGESTIONS:
"""
    return instructions + json.dumps({
        "turns": blind_turns,
        "existing_atoms": existing_atoms,
        "possible_missing_ideas_from_a_separate_AI_audit": audit_suggestions,
    }, ensure_ascii=False)


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


def _validate(value: Any, turns: list[dict[str, str]], existing_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    additions, note = value.get("additions"), value.get("note")
    if not isinstance(additions, list) or len(additions) > 10:
        raise ValueError("additions must be a list of at most 10 atoms")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("note must be a nonempty string")
    available = {turn["turn_id"]: turn["text"] for turn in turns}
    ids = set(existing_ids)
    checked = []
    for index, atom in enumerate(additions):
        field = f"additions[{index}]"
        if not isinstance(atom, dict):
            raise ValueError(f"{field} must be an object")
        for name in ("idea_id", "concept", "summary"):
            if not isinstance(atom.get(name), str) or not atom[name].strip():
                raise ValueError(f"{field}.{name} must be nonempty text")
        idea_id = atom["idea_id"]
        if not _ID.fullmatch(idea_id) or idea_id in ids:
            raise ValueError(f"{field}.idea_id must be unique and not collide with existing IDs")
        ids.add(idea_id)
        if atom.get("kind") not in _KINDS or atom.get("stance") not in _STANCES:
            raise ValueError(f"{field} has unsupported kind or stance")
        if atom.get("prominence") not in {"central", "repeated", "secondary", "brief"}:
            raise ValueError(f"{field}.prominence is unsupported")
        if atom.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"{field}.confidence is unsupported")
        if not isinstance(atom.get("forecastable"), bool):
            raise ValueError(f"{field}.forecastable must be boolean")
        evidence = atom.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4:
            raise ValueError(f"{field}.evidence must contain 1–4 citations")
        checked_evidence = []
        for citation in evidence:
            if not isinstance(citation, dict):
                raise ValueError(f"{field}.evidence item must be an object")
            turn_id, quote = citation.get("turn_id"), citation.get("exact_quote")
            if turn_id not in available or not isinstance(quote, str) or quote not in available[turn_id]:
                raise ValueError(f"{field} has an invalid exact-turn quote")
            checked_evidence.append({"turn_id": turn_id, "exact_quote": quote})
        checked.append({
            "idea_id": idea_id, "concept": atom["concept"].strip(), "summary": atom["summary"].strip(),
            "kind": atom["kind"], "stance": atom["stance"], "prominence": atom["prominence"],
            "confidence": atom["confidence"], "forecastable": atom["forecastable"], "evidence": checked_evidence,
        })
    return {"additions": checked, "note": note.strip()}


def _call_pi(prompt: str, *, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".prompt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        result = subprocess.run([
            executable, "--provider", PROVIDER, "--model", MODEL, "--print", "--mode", "text",
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


def _write_new(path: Path, content: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def generate(
    *, events_path: Path, base_path: Path, audit_path: Path, output_path: Path, timeout: int = 240,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    base_rows, base_bytes = _read_jsonl(base_path)
    audit_rows, audit_bytes = _read_jsonl(audit_path)
    base_hash = _sha256(base_bytes)
    audit_manifest_path = Path(str(audit_path) + ".manifest.json")
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("candidate_atom_artifact_sha256") != base_hash:
        raise ValueError("audit manifest does not refer to the supplied base atom artifact")
    if audit_manifest.get("output_sha256") != _sha256(audit_bytes):
        raise ValueError("quality audit artifact hash does not match its manifest")
    if audit_manifest.get("events_sha256") != _sha256(event_bytes):
        raise ValueError("quality audit was run on a different event corpus")
    event_by_id = {row.get("event_id"): row for row in events}
    base_by_id = {row.get("event_id"): row for row in base_rows}
    audit_by_id = {row.get("event_id"): row for row in audit_rows}
    if len(event_by_id) != len(events) or len(base_by_id) != len(base_rows) or len(audit_by_id) != len(audit_rows):
        raise ValueError("input event IDs must be unique")
    targets = [row for row in audit_rows if row.get("audit_status") == "audited" and row.get("event_completeness", 4) < 4]
    if not targets:
        raise ValueError("no audited events below completeness 4")
    started = datetime.now(timezone.utc).isoformat()
    prompt_hash = _sha256((_SYSTEM_PROMPT + "\n" + _prompt([], [], [])).encode("utf-8"))
    output_rows = []
    for index, audit_row in enumerate(targets, 1):
        event_id = audit_row["event_id"]
        event, base = event_by_id.get(event_id), base_by_id.get(event_id)
        if event is None or base is None:
            raise ValueError(f"missing source or base record for {event_id}")
        if base.get("source_record_sha256") != _canonical_hash(event) or audit_row.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source-record hash mismatch for {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        transcript_hash = _sha256(transcript.encode("utf-8"))
        if base.get("target_transcript_sha256") != transcript_hash or audit_row.get("target_transcript_sha256") != transcript_hash:
            raise ValueError(f"transcript hash mismatch for {event_id}")
        case_id = f"augment-case-{index:04d}"
        blind_turns, blind_to_source, source_to_blind = _blind(turns, case_id)
        existing_ids = {atom["idea_id"] for atom in base.get("atoms", [])}
        existing = []
        turn_text = {turn["turn_id"]: turn["text"] for turn in turns}
        for atom in base.get("atoms", []):
            citations = []
            for citation in atom.get("evidence", []):
                turn_id, quote = citation.get("turn_id"), citation.get("exact_quote")
                if turn_id not in turn_text or not isinstance(quote, str) or quote not in turn_text[turn_id]:
                    raise ValueError(f"base atom has invalid evidence in {event_id}:{atom.get('idea_id')}")
                citations.append({"turn_id": source_to_blind[turn_id], "exact_quote": quote})
            existing.append({
                key: atom[key] for key in ("idea_id", "concept", "summary", "kind", "stance", "forecastable") if key in atom
            } | {"evidence": citations})
        prompt = _prompt(blind_turns, existing, audit_row.get("major_omissions", []))
        errors, validated, response_hash = [], None, None
        for attempt in range(2):
            try:
                response = _call_pi(prompt, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                blind_result = _validate(_parse_object(response), blind_turns, existing_ids)
                additions = []
                for atom in blind_result["additions"]:
                    additions.append({
                        **{key: value for key, value in atom.items() if key != "evidence"},
                        "evidence": [{"turn_id": blind_to_source[item["turn_id"]], "exact_quote": item["exact_quote"]} for item in atom["evidence"]],
                    })
                validated = {"additions": additions, "note": blind_result["note"]}
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
                if attempt == 0:
                    prompt += "\n\nFORMAT REPAIR: Return valid JSON with only genuinely missing, transcript-supported atoms. Do not add filler or duplicate an existing idea."
        output_rows.append({
            "schema_version": 1,
            "artifact_type": "targeted-transcript-idea-atom-supplement",
            "augmentation_version": _PROMPT_VERSION,
            "event_id": event_id,
            "event_date": event.get("event_date"),
            "source_record_sha256": _canonical_hash(event),
            "target_transcript_sha256": transcript_hash,
            "base_atom_artifact_sha256": base_hash,
            "quality_audit_artifact_sha256": _sha256(audit_bytes),
            "auditor_completeness": audit_row["event_completeness"],
            "auditor_suggestions": audit_row.get("major_omissions", []),
            "augmentation_model": {"provider": PROVIDER, "model": MODEL},
            "status": "model_error" if validated is None else ("additions_found" if validated["additions"] else "no_additions"),
            "additions": [] if validated is None else validated["additions"],
            "note": "" if validated is None else validated["note"],
            "response_sha256": response_hash,
            "attempts": len(errors) + 1,
            "errors": errors,
        })
        count = 0 if validated is None else len(validated["additions"])
        print(f"[{index}/{len(targets)}] {case_id}: {output_rows[-1]['status']} ({count} additions)", flush=True)
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite supplement: {output_path}")
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "targeted-transcript-idea-atom-augmentation-run",
        "created_at": started,
        "prompt_version": _PROMPT_VERSION,
        "system_and_prompt_template_sha256": prompt_hash,
        "events_sha256": _sha256(event_bytes),
        "base_atom_artifact_sha256": base_hash,
        "quality_audit_artifact_sha256": _sha256(audit_bytes),
        "audited_event_count": len(targets),
        "successful_event_count": sum(row["status"] != "model_error" for row in output_rows),
        "model_error_count": sum(row["status"] == "model_error" for row in output_rows),
        "event_count_with_additions": sum(row["status"] == "additions_found" for row in output_rows),
        "addition_count": sum(len(row["additions"]) for row in output_rows),
        "provider": PROVIDER,
        "model": MODEL,
        "outputs_local_only": True,
        "output_sha256": _sha256(output_bytes),
        "limitations": [
            "The suggestions came from an AI audit and may be incomplete or false; the augmentation model is instructed to reject unsupported or merely detailed suggestions.",
            "This is targeted post-hoc repair of the same 49 development events, not independent evaluation data.",
            "Assistant-generated atoms are not human-validated semantic gold; any score improvement is diagnostic and may reflect shared model biases.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new(output_path, output_bytes)
    _write_new(manifest_path, manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "addition_count": manifest["addition_count"], "model_errors": manifest["model_error_count"]}


def merge(
    *, events_path: Path, base_path: Path, supplement_path: Path, output_path: Path,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    base_rows, base_bytes = _read_jsonl(base_path)
    supplement_rows, supplement_bytes = _read_jsonl(supplement_path)
    base_hash, supplement_hash = _sha256(base_bytes), _sha256(supplement_bytes)
    supplement_manifest_path = Path(str(supplement_path) + ".manifest.json")
    supplement_manifest_bytes = supplement_manifest_path.read_bytes()
    supplement_manifest = json.loads(supplement_manifest_bytes)
    if supplement_manifest.get("base_atom_artifact_sha256") != base_hash:
        raise ValueError("supplement was generated from a different base atom artifact")
    if supplement_manifest.get("output_sha256") != supplement_hash:
        raise ValueError("supplement artifact hash does not match its manifest")
    event_by_id = {row.get("event_id"): row for row in events}
    supplement_by_id = {row.get("event_id"): row for row in supplement_rows}
    if len(event_by_id) != len(events) or len({row.get("event_id") for row in base_rows}) != len(base_rows):
        raise ValueError("event IDs must be unique")
    if len(supplement_by_id) != len(supplement_rows):
        raise ValueError("supplement event IDs must be unique")
    output_rows, added_count, changed_events = [], 0, 0
    base_by_id = {row["event_id"]: row for row in base_rows}
    for event_id, base in base_by_id.items():
        event = event_by_id.get(event_id)
        if event is None or base.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source-record hash mismatch for {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        transcript_hash = _sha256(transcript.encode("utf-8"))
        if base.get("target_transcript_sha256") != transcript_hash:
            raise ValueError(f"transcript hash mismatch for {event_id}")
        supplement = supplement_by_id.get(event_id)
        if supplement is None:
            output_rows.append(base)
            continue
        if supplement.get("status") == "model_error":
            raise ValueError(f"cannot merge failed supplement for {event_id}")
        if (
            supplement.get("base_atom_artifact_sha256") != base_hash
            or supplement.get("target_transcript_sha256") != transcript_hash
            or supplement.get("source_record_sha256") != _canonical_hash(event)
            or supplement.get("quality_audit_artifact_sha256") != supplement_manifest.get("quality_audit_artifact_sha256")
        ):
            raise ValueError(f"supplement provenance mismatch for {event_id}")
        existing_ids = {atom["idea_id"] for atom in base.get("atoms", [])}
        additions = supplement.get("additions")
        if not isinstance(additions, list):
            raise ValueError(f"invalid additions in {event_id}")
        for atom in additions:
            if atom.get("idea_id") in existing_ids:
                raise ValueError(f"addition ID collides with base for {event_id}")
            existing_ids.add(atom.get("idea_id"))
            for citation in atom.get("evidence", []):
                turn_id = citation.get("turn_id")
                quote = citation.get("exact_quote")
                turn_text = next((turn["text"] for turn in turns if turn["turn_id"] == turn_id), None)
                if turn_text is None or not isinstance(quote, str) or quote not in turn_text:
                    raise ValueError(f"addition has invalid evidence for {event_id}:{atom.get('idea_id')}")
        if additions:
            changed_events += 1
            added_count += len(additions)
        output = dict(base)
        output["atoms"] = [*base.get("atoms", []), *additions]
        output["augmentation"] = {
            "version": _PROMPT_VERSION,
            "status": supplement["status"],
            "added_atom_count": len(additions),
            "added_atoms_response_sha256": supplement.get("response_sha256"),
            "base_artifact_sha256": base_hash,
            "quality_audit_artifact_sha256": supplement.get("quality_audit_artifact_sha256"),
            "provider": supplement.get("augmentation_model", {}).get("provider"),
            "model": supplement.get("augmentation_model", {}).get("model"),
            "ai_audit_suggestions_are_unverified": True,
        }
        output_rows.append(output)
    extra_supplements = set(supplement_by_id) - set(base_by_id)
    if extra_supplements:
        raise ValueError(f"supplement contains unknown events: {sorted(extra_supplements)}")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite merged artifact: {output_path}")
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "augmented-transcript-idea-atom-inventory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events_sha256": _sha256(event_bytes),
        "base_atom_artifact_sha256": base_hash,
        "supplement_artifact_sha256": supplement_hash,
        "supplement_manifest_sha256": _sha256(supplement_manifest_bytes),
        "event_count": len(output_rows),
        "event_count_with_additions": changed_events,
        "added_atom_count": added_count,
        "total_atom_count": sum(len(row.get("atoms", [])) for row in output_rows),
        "outputs_local_only": True,
        "limitations": [
            "Original atoms are preserved; additions are assistant-generated candidates, not verified corrections.",
            "The artifact is targeted to events selected by one AI auditor's completeness ratings and is not an independently evaluated outcome set.",
        ],
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new(output_path, output_bytes)
    _write_new(manifest_path, manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "event_count_with_additions": changed_events, "added_atom_count": added_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="create a targeted supplement")
    generate_parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    generate_parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    generate_parser.add_argument("--audit", type=Path, default=AUDIT_DEFAULT)
    generate_parser.add_argument("--output", type=Path, default=SUPPLEMENT_DEFAULT)
    generate_parser.add_argument("--timeout", type=int, default=240)
    merge_parser = subparsers.add_parser("merge", help="merge supplement into a new immutable inventory")
    merge_parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    merge_parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    merge_parser.add_argument("--supplement", type=Path, default=SUPPLEMENT_DEFAULT)
    merge_parser.add_argument("--output", type=Path, default=MERGED_DEFAULT)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(events_path=args.events, base_path=args.base, audit_path=args.audit, output_path=args.output, timeout=args.timeout)
    else:
        result = merge(events_path=args.events, base_path=args.base, supplement_path=args.supplement, output_path=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
