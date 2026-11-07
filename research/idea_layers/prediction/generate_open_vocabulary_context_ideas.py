"""Opt-in Codex candidate generator using only pre-event headline titles.

This is a research comparator to the frozen closed-vocabulary classifier. It
emits open-vocabulary idea candidates and cited headline evidence, not calibrated
probabilities or a replacement for the frozen baseline.
"""

from __future__ import annotations

import argparse
from datetime import date
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from speechprefix.context_theme_ranker import read_jsonl
from speechprefix.headline_idea_matching import _target_turns, extract_transcript_ideas

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
SNAPSHOTS_DEFAULT = ROOT / "data/rollcall-context-snapshots-retro-combined-v1.jsonl"
ENGINE_VERSION = "codex-open-vocabulary-context-ideas-v1"
PROMPT_VERSION = "headline-only-idea-candidates-v1"
PROVIDER = "openai-codex"
MODEL = "gpt-6.1-sol"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/predictions/codex-open-vocabulary-context-ideas-v1.jsonl"
_SYSTEM_PROMPT = (
    "You are a cautious retrospective research forecaster. Use only the supplied pre-event headline titles. "
    "Do not use tools, outside knowledge, event identity, target transcripts, or information after the cutoff. "
    "Return one JSON object and no markdown."
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prompt(headlines: list[dict[str, str]]) -> str:
    return f"""Generate a ranked, open-vocabulary list of concrete speech-idea candidates suggested by the supplied headline titles.

Rules:
- Predict substantive ideas the speaker might address, not generic themes such as politics, the economy, or foreign affairs.
- Preserve named entities, policies, conflicts, and distinctions when the headlines support them. Do not merge different countries, people, or policy mechanisms.
- Generate a candidate only when at least one supplied headline directly supports it. Do not fill a quota; abstention/short lists are allowed.
- Each candidate must cite one or more supplied `headline_id` values. The cited title(s) are evidence, not proof the idea will be mentioned.
- Do not infer the target speaker's position, promised action, or predicted wording unless the headline states it.
- Return at most 15 candidates. Do not give probabilities or confidence scores. Rank by strength and specificity of the supplied evidence.

Return exactly {{"candidates": [{{"idea": "concise concrete idea phrase", "headline_ids": ["supplied ID"], "rationale": "brief title-grounded rationale"}}]}}.

PRE-EVENT HEADLINE TITLES:
{json.dumps(headlines, ensure_ascii=False)}"""


def _parse_candidates(text: str, allowed_ids: set[str]) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    value = None
    for index, character in enumerate(text):
        if text[index] != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if not isinstance(value, dict) or set(value) != {"candidates"} or not isinstance(value["candidates"], list):
        raise ValueError("response must be an object containing only a candidates list")
    if len(value["candidates"]) > 15:
        raise ValueError("response exceeds 15 candidates")
    checked, seen = [], set()
    for index, candidate in enumerate(value["candidates"]):
        if not isinstance(candidate, dict) or set(candidate) != {"idea", "headline_ids", "rationale"}:
            raise ValueError(f"candidate {index} must contain exactly idea, headline_ids, rationale")
        idea, ids, rationale = candidate["idea"], candidate["headline_ids"], candidate["rationale"]
        if not isinstance(idea, str) or not idea.strip() or len(idea) > 240:
            raise ValueError(f"candidate {index} idea must be nonempty text of at most 240 characters")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or item not in allowed_ids for item in ids):
            raise ValueError(f"candidate {index} must cite one or more supplied headline IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"candidate {index} repeats a headline ID")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
            raise ValueError(f"candidate {index} rationale must be nonempty and at most 500 characters")
        key = " ".join(idea.casefold().split())
        if key in seen:
            raise ValueError("candidate ideas must be unique")
        seen.add(key)
        checked.append({"rank": index + 1, "idea": idea.strip(), "headline_ids": ids, "rationale": rationale.strip()})
    return checked


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".codex-idea-prompt", delete=False) as handle:
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
            if any(item in output for item in ("insufficient balance", "invalid api key", "unauthorized", "authentication failed", "forbidden")):
                raise RuntimeError("Codex provider rejected the idea-generation request")
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty idea-generation response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def run(
    *, events_path: Path, snapshots_path: Path, output_path: Path,
    event_ids: set[str], provider: str = PROVIDER, model: str = MODEL, timeout: int = 300,
) -> dict[str, Any]:
    if not event_ids:
        raise ValueError("provide explicit --event-id values; this research run is bounded")
    events = read_jsonl(events_path)
    snapshots = read_jsonl(snapshots_path)
    event_by_id = {row.get("event_id"): row for row in events}
    snapshot_by_id = {row.get("event_id"): row for row in snapshots}
    if len(event_by_id) != len(events) or len(snapshot_by_id) != len(snapshots):
        raise ValueError("duplicate event IDs in input corpus")
    if event_ids - set(event_by_id) or event_ids - set(snapshot_by_id):
        raise ValueError("every selected event must have both a source transcript record and context snapshot")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite Codex idea artifact: {output_path}")

    rows = []
    for index, event_id in enumerate(sorted(event_ids), 1):
        event, snapshot = event_by_id[event_id], snapshot_by_id[event_id]
        event_day = date.fromisoformat(str(event.get("event_date", "")))
        if snapshot.get("event_date") != event_day.isoformat() or snapshot.get("exploratory_only") is not True:
            raise ValueError(f"snapshot date/status mismatch for {event_id}")
        source_rows = snapshot.get("candidate_sources")
        if not isinstance(source_rows, list):
            raise ValueError(f"candidate_sources must be a list for {event_id}")
        eligible, seen, source_by_blind_id = [], set(), {}
        start = date.fromordinal(event_day.toordinal() - 7)
        for source_index, source in enumerate(source_rows):
            title, source_id = source.get("title"), source.get("source_id") or source.get("source_url") or source.get("url")
            try:
                published = date.fromisoformat(str(source.get("published_at", ""))[:10])
            except (TypeError, ValueError):
                continue
            if published < start or published >= event_day:
                continue
            if not isinstance(title, str) or not title.strip() or not isinstance(source_id, str) or not source_id:
                continue
            if source_id in seen:
                continue
            seen.add(source_id)
            blind_id = f"h{len(eligible)+1:03d}"
            eligible.append({"headline_id": blind_id, "title": title.strip()})
            source_by_blind_id[blind_id] = {
                "source_id": source_id,
                "published_at": published.isoformat(),
                "title": title.strip(),
                "publisher": str(source.get("publisher", "")).strip(),
            }
        prompt = _prompt(eligible)
        response_hash = None
        candidates = None
        errors = []
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                candidates = _parse_candidates(response, {item["headline_id"] for item in eligible})
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, RuntimeError) and "provider rejected" in str(exc):
                    raise
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < 2:
                    prompt += "\n\nFORMAT REPAIR: Use only supplied headline IDs; return exactly the required JSON and no unsupported claims.\nPREVIOUS RESPONSE:\n" + response
        status = "generated" if candidates is not None else "model_error"
        target_ideas = extract_transcript_ideas(_target_turns(event))
        target_ids = sorted({idea["idea_id"] for idea in target_ideas})
        candidates = candidates or []
        rows.append({
            "event_id": event_id,
            "event_date": event_day.isoformat(),
            "status": status,
            "provider": provider,
            "model": model,
            "engine_version": ENGINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "context_cutoff_exclusive": event_day.isoformat(),
            "eligible_headline_count": len(eligible),
            "eligible_headlines_sha256": _canonical_hash(eligible),
            "headline_evidence": source_by_blind_id,
            "candidates": candidates,
            "weak_target_ids_posthoc_only": target_ids,
            "errors": errors,
            "response_sha256": response_hash,
        })
        print(f"[{index}/{len(event_ids)}] case-{index:03d}: {status}, headlines={len(eligible)}, candidates={len(candidates)}", flush=True)
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "codex-open-vocabulary-context-idea-generation-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "events_sha256": _sha256(events_path.read_bytes()),
        "snapshots_sha256": _sha256(snapshots_path.read_bytes()),
        "event_ids": sorted(event_ids),
        "event_count": len(rows),
        "success_count": sum(row["status"] == "generated" for row in rows),
        "candidate_count": sum(len(row["candidates"]) for row in rows),
        "target_transcripts_used_in_prompt": False,
        "transcript_labels_used_only_posthoc_for_scoring": True,
        "semantic_accuracy_validated": False,
        "outputs_local_only": True,
        "limitations": [
            "Headline snapshots were retrieved retrospectively; publication dates do not prove historical point-in-time availability.",
            "Weak exact phrase matches are incomplete lexical labels, not semantic gold or forecast accuracy.",
            "The model emits evidence-grounded candidates but no calibrated probabilities.",
        ],
        "output_sha256": _sha256(output_bytes),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_bytes)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output_path), "manifest": str(manifest_path), "events": len(rows), "successes": manifest["success_count"], "candidates": manifest["candidate_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--event-id", action="append", required=True)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events, snapshots_path=args.snapshots, output_path=args.output,
        event_ids=set(args.event_id), provider=args.provider, model=args.model, timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
