"""Cross-model, transcript-only quality audit for semantic idea-atom artifacts."""

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

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
ATOMS_DEFAULT = ROOT / "data/source-archives/2019-weak-empty-semantic-idea-atoms-v1-complete.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/legacy_atoms/2019-semantic-idea-atom-quality-audit-v2.jsonl"
AUDITOR_PROVIDER = "openai-codex"
AUDITOR_MODEL = "gpt-6.1-sol"
_PROMPT_VERSION = "transcript-only-cross-model-atom-audit-v2-no-confidence-prominence"
_SYSTEM_PROMPT = (
    "You are a skeptical transcript-only quality auditor. Review the complete supplied target-speaker transcript and candidate idea atoms. "
    "Use only this transcript and the supplied atom evidence; do not use outside knowledge, event identity, headlines, forecasts, or prior labels. "
    "Return exactly one JSON object and no markdown."
)
_FLAGS = {
    "unsupported", "overstated", "understated", "conflates_ideas", "over_split", "duplicate",
    "vague", "stance_mismatch", "kind_mismatch", "forecastability_mismatch", "missing_key_idea",
}


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
                raise ValueError("target-speaker transcript turn is missing a turn_id")
            turns.append({"turn_id": turn_id, "text": text.strip()})
    return turns


def _blind(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    blind_turns, mapping = [], {}
    for index, turn in enumerate(turns):
        blind_id = f"{case_id}:turn:{index:04d}"
        mapping[blind_id] = turn["turn_id"]
        blind_turns.append({"turn_id": blind_id, "text": turn["text"]})
    return blind_turns, mapping


def _prompt(blind_turns: list[dict[str, str]], atoms: list[dict[str, Any]]) -> str:
    task = """Audit these candidate atoms against the complete transcript. Do not rewrite them. Rate each atom on four 0–4 scales:
- evidence_support: 4 = directly and fully supported by its cited quote(s); 3 = substantially supported with a small inference; 2 = only partly supported or materially ambiguous; 1 = weak support; 0 = unsupported or contradicted.
- atomicity: 4 = one distinct, appropriately scoped idea; 3 = minor over/under-grouping; 2 = noticeable merge or split problem; 1 = serious granularity problem; 0 = not a coherent atom.
- summary_stance_fidelity: 4 = accurate neutral summary and stance; 3 = minor nuance error; 2 = meaningful over/understatement; 1 = major mismatch; 0 = contradicts the transcript.
- type_forecastability_fidelity: 4 = kind and forecastable flag fit the transcript content and task; 3 = minor uncertainty; 2 = debatable; 1 = likely wrong; 0 = clearly wrong.

Also rate event_completeness 0–4: 4 = the inventory captures the salient substantive ideas without an obvious major omission; 3 = one modest omission; 2 = a significant omission; 1 = multiple major omissions; 0 = it misses nearly all salient ideas. Do not penalize a transcript that truly contains only a brief ceremonial point. Do not invent omissions based on outside event knowledge.

Return JSON only with this shape:
{
  "event_completeness": 0,
  "major_omissions": ["short transcript-grounded concept"],
  "atom_reviews": [
    {
      "idea_id": "the supplied ID, unchanged",
      "evidence_support": 0,
      "atomicity": 0,
      "summary_stance_fidelity": 0,
      "type_forecastability_fidelity": 0,
      "issue_flags": [],
      "reason": "brief explanation grounded in cited text"
    }
  ],
  "overall_note": "brief transcript-grounded assessment"
}

Use only these issue_flags: unsupported, overstated, understated, conflates_ideas, over_split, duplicate, vague, stance_mismatch, kind_mismatch, forecastability_mismatch, missing_key_idea. Keep reasons short. Return exactly one atom_review per supplied atom and do not silently omit an atom. Each cited blind turn ID is not an event identifier.

TRANSCRIPT TURNS:
"""
    return task + json.dumps({"turns": blind_turns, "candidate_atoms": atoms}, ensure_ascii=False)


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
    raise ValueError("auditor response did not contain a JSON object")


def _validate(value: Any, atom_ids: set[str], blind_turn_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("audit response must be an object")
    completeness = value.get("event_completeness")
    if not isinstance(completeness, int) or isinstance(completeness, bool) or not 0 <= completeness <= 4:
        raise ValueError("event_completeness must be an integer from 0 to 4")
    omissions, reviews = value.get("major_omissions"), value.get("atom_reviews")
    if not isinstance(omissions, list) or len(omissions) > 12 or any(not isinstance(x, str) or not x.strip() for x in omissions):
        raise ValueError("major_omissions must be a list of nonempty strings")
    if not isinstance(reviews, list):
        raise ValueError("atom_reviews must be a list")
    by_id = {}
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("idea_id"), str):
            raise ValueError("each atom review needs an idea_id")
        idea_id = review["idea_id"]
        if idea_id not in atom_ids or idea_id in by_id:
            raise ValueError("atom review IDs must exactly match unique supplied IDs")
        scores = {}
        for field in ("evidence_support", "atomicity", "summary_stance_fidelity", "type_forecastability_fidelity"):
            score = review.get(field)
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                raise ValueError(f"{idea_id}.{field} must be an integer from 0 to 4")
            scores[field] = score
        flags = review.get("issue_flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) or flag not in _FLAGS for flag in flags):
            raise ValueError(f"{idea_id}.issue_flags contains an unsupported value")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{idea_id}.reason must be nonempty")
        by_id[idea_id] = {"idea_id": idea_id, **scores, "issue_flags": flags, "reason": reason.strip()}
    if set(by_id) != atom_ids:
        raise ValueError("auditor omitted one or more candidate atoms")
    note = value.get("overall_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("overall_note must be nonempty")
    return {
        "event_completeness": completeness,
        "major_omissions": [item.strip() for item in omissions],
        "atom_reviews": [by_id[idea_id] for idea_id in sorted(by_id)],
        "overall_note": note.strip(),
    }


def _prepare_atoms(
    atoms: list[dict[str, Any]],
    *,
    source_to_blind: dict[str, str],
    original_turns: dict[str, str],
) -> tuple[list[dict[str, Any]], set[str], int]:
    audit_atoms, atom_ids, evidence_total = [], set(), 0
    for atom in atoms:
        atom_id = atom.get("idea_id")
        if not isinstance(atom_id, str) or atom_id in atom_ids:
            raise ValueError("invalid/duplicate atom ID")
        atom_ids.add(atom_id)
        audit_atom = {key: value for key, value in atom.items() if key not in {"evidence", "confidence", "prominence"}}
        blind_evidence = []
        for item in atom.get("evidence", []):
            turn_id, quote = item.get("turn_id"), item.get("exact_quote")
            evidence_total += 1
            if turn_id not in original_turns or not isinstance(quote, str) or quote not in original_turns[turn_id]:
                raise ValueError(f"invalid evidence quote for {atom_id}")
            blind_evidence.append({"turn_id": source_to_blind[turn_id], "exact_quote": quote})
        audit_atom["evidence"] = blind_evidence
        audit_atoms.append(audit_atom)
    return audit_atoms, atom_ids, evidence_total


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required for this audit")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".audit-prompt", delete=False) as handle:
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


def _event_score(review: dict[str, Any], *, citation_integrity: float) -> float:
    atoms = review["atom_reviews"]
    means = {
        name: sum(atom[name] for atom in atoms) / len(atoms)
        for name in ("evidence_support", "atomicity", "summary_stance_fidelity", "type_forecastability_fidelity")
    } if atoms else {name: 0.0 for name in ("evidence_support", "atomicity", "summary_stance_fidelity", "type_forecastability_fidelity")}
    return (
        0.35 * means["evidence_support"] / 4 * 100
        + 0.20 * means["atomicity"] / 4 * 100
        + 0.20 * means["summary_stance_fidelity"] / 4 * 100
        + 0.10 * means["type_forecastability_fidelity"] / 4 * 100
        + 0.10 * review["event_completeness"] / 4 * 100
        + 0.05 * citation_integrity
    )


def run(
    *, events_path: Path, atoms_path: Path, output_path: Path,
    provider: str = AUDITOR_PROVIDER, model: str = AUDITOR_MODEL, timeout: int = 240,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    event_by_id = {row.get("event_id"): row for row in events}
    if len(event_by_id) != len(events):
        raise ValueError("event corpus contains duplicate IDs")
    atom_rows, atom_bytes = _read_jsonl(atoms_path)
    if len({row.get("event_id") for row in atom_rows}) != len(atom_rows):
        raise ValueError("atom artifact contains duplicate event IDs")
    audited = []
    prompt_template_hash = _sha256((_SYSTEM_PROMPT + "\n" + _prompt([], [])).encode("utf-8"))
    for index, atom_row in enumerate(atom_rows, 1):
        event_id = atom_row.get("event_id")
        event = event_by_id.get(event_id)
        if event is None:
            raise ValueError(f"event {event_id!r} is missing from source corpus")
        if atom_row.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source record hash mismatch for {event_id}")
        extraction_model = atom_row.get("reviewer", {}).get("model")
        if not isinstance(extraction_model, str) or extraction_model == model:
            raise ValueError("auditor must differ from the recorded extraction model")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        if atom_row.get("target_transcript_sha256") != _sha256(transcript.encode("utf-8")):
            raise ValueError(f"transcript hash mismatch for {event_id}")
        original_turns = {turn["turn_id"]: turn["text"] for turn in turns}
        case_id = f"quality-case-{index:04d}"
        blind_turns, turn_mapping = _blind(turns, case_id)
        source_to_blind = {source_id: blind_id for blind_id, source_id in turn_mapping.items()}
        audit_atoms, atom_ids, evidence_total = _prepare_atoms(
            atom_row.get("atoms", []),
            source_to_blind=source_to_blind,
            original_turns=original_turns,
        )
        exact_quotes = evidence_total
        prompt = _prompt(blind_turns, audit_atoms)
        errors = []
        result = None
        response_hash = None
        for attempt in range(2):
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                result = _validate(_parse_object(response), atom_ids, {turn["turn_id"] for turn in blind_turns})
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
                if attempt == 0:
                    prompt += "\n\nFORMAT REPAIR: Correct the response to include every supplied atom ID exactly once and all required 0–4 integer ratings. Do not change the transcript or candidate atoms."
        if result is None:
            audited.append({
                "event_id": event_id, "event_date": event.get("event_date"),
                "extraction_model": extraction_model, "audit_status": "model_error",
                "errors": errors, "atoms": [], "event_completeness": None,
                "citation_integrity_percent": 100.0 if evidence_total == exact_quotes else 0.0,
            })
            print(f"[{index}/{len(atom_rows)}] quality-case-{index:04d}: model_error", flush=True)
            continue
        mapped_reviews = [
            {**review, "evidence": [
                {"turn_id": turn_mapping[item["turn_id"]], "exact_quote": item["exact_quote"]}
                for item in next(atom for atom in audit_atoms if atom["idea_id"] == review["idea_id"])["evidence"]
            ]}
            for review in result["atom_reviews"]
        ]
        citation_integrity = 100.0 * exact_quotes / evidence_total if evidence_total else 100.0
        event_review = {**result, "atom_reviews": result["atom_reviews"]}
        audited.append({
            "event_id": event_id,
            "event_date": event.get("event_date"),
            "source_record_sha256": atom_row["source_record_sha256"],
            "target_transcript_sha256": atom_row["target_transcript_sha256"],
            "extraction_model": extraction_model,
            "auditor_model": model,
            "audit_prompt_version": _PROMPT_VERSION,
            "audit_status": "audited",
            "atom_count": len(mapped_reviews),
            "evidence_quote_count": evidence_total,
            "citation_integrity_percent": round(citation_integrity, 4),
            "event_completeness": result["event_completeness"],
            "event_score_100": round(_event_score(event_review, citation_integrity=citation_integrity), 2),
            "major_omissions": result["major_omissions"],
            "atom_reviews": mapped_reviews,
            "overall_note": result["overall_note"],
            "auditor_response_sha256": response_hash,
            "auditor_attempts": len(errors) + 1,
            "auditor_errors": errors,
        })
        print(f"[{index}/{len(atom_rows)}] quality-case-{index:04d}: audited ({len(mapped_reviews)} atoms)", flush=True)
    good = [row for row in audited if row["audit_status"] == "audited"]
    dimensions = ("evidence_support", "atomicity", "summary_stance_fidelity", "type_forecastability_fidelity")
    nonempty_good = [row for row in good if row["atom_reviews"]]
    per_event_dimension_means = {
        name: sum(
            sum(atom[name] for atom in row["atom_reviews"]) / len(row["atom_reviews"])
            for row in nonempty_good
        ) / len(nonempty_good) if nonempty_good else 0.0
        for name in dimensions
    }
    event_completeness_mean = sum(row["event_completeness"] for row in good) / len(good) if good else 0.0
    citation_integrity_mean = sum(row["citation_integrity_percent"] for row in audited) / len(audited) if audited else 0.0
    mean_score = sum(row["event_score_100"] for row in good) / len(good) if good else 0.0
    sorted_scores = sorted(row["event_score_100"] for row in good)
    median_score = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0
    atom_reviews = [atom for row in good for atom in row["atom_reviews"]]
    flag_counts = {}
    for atom in atom_reviews:
        for flag in atom["issue_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite quality audit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audited).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "transcript-idea-atom-cross-model-quality-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events_sha256": _sha256(event_bytes),
        "candidate_atom_artifact_sha256": _sha256(atom_bytes),
        "prompt_version": _PROMPT_VERSION,
        "prompt_template_sha256": prompt_template_hash,
        "auditor_provider": provider,
        "auditor_model": model,
        "audited_event_count": len(atom_rows),
        "successful_audit_count": len(good),
        "audit_error_count": len(audited) - len(good),
        "candidate_atom_count": sum(row.get("atom_count", 0) for row in good),
        "exact_quote_count": sum(row.get("evidence_quote_count", 0) for row in good),
        "event_weighted_dimension_means_0_to_4": per_event_dimension_means,
        "event_completeness_mean_0_to_4": event_completeness_mean,
        "exact_citation_integrity_percent": citation_integrity_mean,
        "quality_score_100_event_macro_mean": round(mean_score, 2),
        "quality_score_100_event_macro_median": round(median_score, 2),
        "issue_flag_counts": flag_counts,
        "scoring_rubric_weights": {
            "exact_quote_integrity": 5,
            "evidence_support": 35,
            "atomicity": 20,
            "summary_stance_fidelity": 20,
            "type_forecastability_fidelity": 10,
            "event_completeness": 10,
        },
        "score_interpretation": "Cross-model AI audit diagnostic only; not semantic accuracy, human review, or an objective quality estimate.",
        "limitations": [
            "The auditor is another AI model, not human adjudication or semantic gold; provider/model biases and correlated errors remain possible.",
            "The audited candidates were produced by another model from the same provider family; this is a cross-model check, not an independent benchmark.",
            "Completeness scores and issue flags are judgments, not exhaustive proof of omissions or correctness.",
            "The audit applies only to the 49 events selected for having zero compact-lexicon labels.",
        ],
        "outputs_local_only": True,
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "mean_score_100": round(mean_score, 2),
        "median_score_100": round(median_score, 2),
        "audited_events": len(good),
        "audit_errors": len(audited) - len(good),
        "atom_count": sum(row.get("atom_count", 0) for row in good),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--atoms", type=Path, default=ATOMS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--provider", default=AUDITOR_PROVIDER)
    parser.add_argument("--model", default=AUDITOR_MODEL)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        atoms_path=args.atoms,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
