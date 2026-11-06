"""Model-as-judge diagnostics for separate expression anchors and fine edges.

Verdicts and omission suggestions are interpretive diagnostics, not semantic
gold, precision/recall, or a quality certification.
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
    FINE_RELATIONS,
    canonical_sha256,
    validate_relation_candidate_record,
    validate_relation_error_record,
)
from speechprefix.idea_layer.semantic_anchors import validate_expression_anchor_record
from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
AUDITOR_PROVIDER = "openai-codex"
AUDITOR_MODEL = "gpt-6-astra"
PROMPT_VERSION = "separate-anchor-and-fine-edge-review-v2"
OUTPUT_DEFAULT = ROOT / "data/source-archives/2019-layered-relationship-audit-v3.jsonl"
_SYSTEM_PROMPT = (
    "You are a skeptical transcript-only reviewer of semantic anchors and relation labels. "
    "Use only supplied transcript evidence and frozen nodes. Do not use tools, outside knowledge, event identity, news, or forecasts. "
    "Return one JSON object and no markdown."
)
_ANCHOR_VERDICTS = {"direct", "uncertain", "mismatch"}
_EDGE_VERDICTS = {"plausible", "uncertain", "mismatch"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _blind(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    result, blind_to_source = [], {}
    for index, turn in enumerate(turns):
        blind_id = f"relationship-audit-case-{case_id}:turn:{index:04d}"
        result.append({"turn_id": blind_id, "text": turn["text"]})
        blind_to_source[blind_id] = turn["turn_id"]
    return result, blind_to_source


def _prompt(
    blind_turns: list[dict[str, str]],
    outcome: dict[str, Any],
    anchor_record: dict[str, Any],
    relation_record: dict[str, Any],
    source_to_blind: dict[str, str],
) -> str:
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
    payload = {
        "turns": blind_turns,
        "mentions": mentions,
        "ideas": ideas,
        "expression_anchors": anchor_record["anchors"],
        "ideas_without_anchor": anchor_record["unanchored_idea_ids"],
        "candidate_fine_edges": relation_record.get("edges", []),
        "fine_edge_input_status": (
            "relation extraction failed; an empty candidate list is not evidence of no relations"
            if relation_record.get("status") == "model_error"
            else "validated candidate edges"
        ),
    }
    return f"""Review the two distinct annotation layers. The transcript and quote-backed mentions are the only evidence.

A. **Expression anchors.** For each supplied anchor, decide whether the mention itself directly expresses the idea (`direct`), whether the distinction is unclear (`uncertain`), or whether the mention is only related/supporting/contextual rather than a direct expression (`mismatch`). Do not infer direct expression merely because another mention in the same transcript expresses the idea. Return one review for every anchor ID.

Also scan for up to 20 obvious direct-expression anchor pairs missing from the supplied anchor set. Return only mention/idea IDs that exist in the input, and do not duplicate an existing pair. These are candidate omissions, not proof of incomplete recall.

B. **Fine-grained relation edges.** For each supplied candidate edge, judge whether its typed relation is a plausible reading of the mention→idea connection (`plausible`), ambiguous among relation types (`uncertain`), or a mismatch (`mismatch`). A plausible verdict is not objective truth. Return one review for every edge ID.

Relation definitions:
{json.dumps(FINE_RELATION_DEFINITIONS, ensure_ascii=False, indent=2)}

Do not change nodes, anchors, or edges. Do not score forecastability, use outside context, or claim completeness. Notes should be brief. Missing-anchor suggestions inherit evidence from the cited mention ID; do not invent quotes.

Return exactly this JSON shape:
{{
  "anchor_reviews": [{{"anchor_id": "supplied ID", "verdict": "direct|uncertain|mismatch", "note": "brief reason"}}],
  "missing_anchor_candidates": [{{"mention_id": "supplied ID", "idea_id": "supplied ID", "note": "brief reason"}}],
  "edge_reviews": [{{"edge_id": "supplied ID", "verdict": "plausible|uncertain|mismatch", "note": "brief reason"}}]
}}

AUDIT INPUT:
{json.dumps(payload, ensure_ascii=False)}"""


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
    raise ValueError("review response did not contain a JSON object")


def _review_list(value: Any, *, key: str, expected_ids: set[str], allowed_verdicts: set[str]) -> list[dict[str, str]]:
    items = value.get(key)
    if not isinstance(items, list):
        raise ValueError(f"{key} must be a list")
    indexed = {}
    id_key = "anchor_id" if key == "anchor_reviews" else "edge_id"
    for item in items:
        if not isinstance(item, dict) or set(item) != {id_key, "verdict", "note"}:
            raise ValueError(f"each {key} item must contain exactly {id_key}, verdict, and note")
        identifier, verdict, note = item[id_key], item["verdict"], item["note"]
        if not isinstance(identifier, str) or identifier not in expected_ids or identifier in indexed:
            raise ValueError(f"{key} IDs must be unique and match the supplied candidates")
        if not isinstance(verdict, str) or verdict not in allowed_verdicts:
            raise ValueError(f"{key} verdict is unsupported")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{key} note must be nonempty")
        indexed[identifier] = {id_key: identifier, "verdict": verdict, "note": note.strip()}
    if set(indexed) != expected_ids:
        raise ValueError(f"{key} omitted one or more supplied candidates")
    return [indexed[identifier] for identifier in sorted(indexed)]


def _validate_response(
    value: Any,
    *,
    expected_anchor_ids: set[str],
    expected_edge_ids: set[str],
    mention_ids: set[str],
    idea_ids: set[str],
    existing_anchor_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    expected_keys = {"anchor_reviews", "missing_anchor_candidates", "edge_reviews"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"response must contain exactly: {sorted(expected_keys)}")
    anchor_reviews = _review_list(
        value, key="anchor_reviews", expected_ids=expected_anchor_ids, allowed_verdicts=_ANCHOR_VERDICTS,
    )
    edge_reviews = _review_list(
        value, key="edge_reviews", expected_ids=expected_edge_ids, allowed_verdicts=_EDGE_VERDICTS,
    )
    missing = value["missing_anchor_candidates"]
    if not isinstance(missing, list) or len(missing) > 20:
        raise ValueError("missing_anchor_candidates must be a list of at most 20 objects")
    missing_pairs, checked_missing = set(), []
    for item in missing:
        if not isinstance(item, dict) or set(item) != {"mention_id", "idea_id", "note"}:
            raise ValueError("each missing-anchor candidate must contain mention_id, idea_id, and note")
        mention_id, idea_id, note = item["mention_id"], item["idea_id"], item["note"]
        pair = (mention_id, idea_id) if isinstance(mention_id, str) and isinstance(idea_id, str) else None
        if pair is None or mention_id not in mention_ids or idea_id not in idea_ids:
            raise ValueError("missing-anchor candidate references an unknown node")
        if pair in existing_anchor_pairs or pair in missing_pairs:
            raise ValueError("missing-anchor candidate duplicates an existing or returned pair")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("missing-anchor candidate note must be nonempty")
        missing_pairs.add(pair)
        checked_missing.append({"mention_id": mention_id, "idea_id": idea_id, "note": note.strip()})
    return {
        "anchor_reviews": anchor_reviews,
        "missing_anchor_candidates": checked_missing,
        "edge_reviews": edge_reviews,
    }


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".relation-audit-prompt", delete=False) as handle:
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
                raise RuntimeError("provider rejected the relationship audit for authentication or account balance")
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty review response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def run(
    *,
    events_path: Path,
    outcomes_path: Path,
    relations_path: Path,
    anchors_path: Path,
    output_path: Path,
    provider: str = AUDITOR_PROVIDER,
    model: str = AUDITOR_MODEL,
    timeout: int = 300,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    event_by_id = {event.get("event_id"): event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("event corpus contains duplicate IDs")
    outcomes, outcome_bytes = _read_jsonl(outcomes_path)
    outcome_by_id = {row.get("event_id"): row for row in outcomes}
    if len(outcome_by_id) != len(outcomes):
        raise ValueError("outcome artifact contains duplicate IDs")
    anchor_rows, anchor_bytes = _read_jsonl(anchors_path)
    anchor_by_id = {row.get("event_id"): row for row in anchor_rows}
    if len(anchor_by_id) != len(anchor_rows):
        raise ValueError("expression-anchor artifact contains duplicate IDs")
    relation_rows, relation_bytes = _read_jsonl(relations_path)
    relation_by_id = {row.get("event_id"): row for row in relation_rows}
    if len(relation_by_id) != len(relation_rows):
        raise ValueError("relation candidate artifact contains duplicate IDs")
    if set(relation_by_id) - set(outcome_by_id):
        raise ValueError("relation candidate artifact references an unknown outcome event")
    if set(relation_by_id) - set(anchor_by_id):
        raise ValueError("relation candidate artifact references a missing expression-anchor record")
    selected_anchor_rows = [anchor_by_id[event_id] for event_id in relation_by_id]
    anchor_provenance = {(row.get("extractor", {}).get("provider"), row.get("extractor", {}).get("model")) for row in selected_anchor_rows}
    relation_provenance = {(row.get("extractor", {}).get("provider"), row.get("extractor", {}).get("model")) for row in relation_rows}
    if len(anchor_provenance) != 1 or len(relation_provenance) != 1:
        raise ValueError("audited events must share anchor and relation extractor provenance")
    anchor_extractor = selected_anchor_rows[0]["extractor"]
    relation_extractor = relation_rows[0]["extractor"]

    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite relationship audit: {output_path}")
    results = []
    for index, event_id in enumerate(sorted(relation_by_id), 1):
        if event_id not in event_by_id:
            raise ValueError(f"event missing from transcript corpus: {event_id}")
        event, outcome = event_by_id[event_id], outcome_by_id[event_id]
        anchor_record, relation_record = anchor_by_id[event_id], relation_by_id[event_id]
        if outcome.get("source_record_sha256") != canonical_sha256(event):
            raise ValueError(f"source event hash mismatch: {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        if outcome.get("target_transcript_sha256") != _sha256(transcript.encode("utf-8")):
            raise ValueError(f"target transcript hash mismatch: {event_id}")
        turns_by_id = {turn["turn_id"]: turn["text"] for turn in turns}
        validate_layered_outcome(outcome, turns_by_id)
        validate_expression_anchor_record(anchor_record, outcome, turns_by_id)
        relation_failed = relation_record.get("status") == "model_error"
        if relation_failed:
            validate_relation_error_record(relation_record, outcome, anchor_record, turns_by_id)
        else:
            validate_relation_candidate_record(relation_record, outcome, anchor_record, turns_by_id)
        blind, blind_to_source = _blind(turns, f"{index:04d}")
        source_to_blind = {source_id: blind_id for blind_id, source_id in blind_to_source.items()}
        prompt = _prompt(blind, outcome, anchor_record, relation_record, source_to_blind)
        expected_anchor_ids = {item["anchor_id"] for item in anchor_record["anchors"]}
        expected_edge_ids = {item["edge_id"] for item in relation_record.get("edges", [])}
        existing_pairs = {(item["mention_id"], item["idea_id"]) for item in anchor_record["anchors"]}
        mention_ids = {item["mention_id"] for item in outcome["mentions"]}
        idea_ids = {item["idea_id"] for item in outcome["ideas"]}
        errors, result, response_hash = [], None, None
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                result = _validate_response(
                    _parse_object(response),
                    expected_anchor_ids=expected_anchor_ids,
                    expected_edge_ids=expected_edge_ids,
                    mention_ids=mention_ids,
                    idea_ids=idea_ids,
                    existing_anchor_pairs=existing_pairs,
                )
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, RuntimeError) and "provider rejected" in str(exc):
                    raise
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < 2:
                    prompt += "\n\nFORMAT REPAIR: Return exactly one valid review per supplied anchor and edge ID, plus at most 20 valid missing-anchor pairs. Do not add unknown IDs.\nPREVIOUS RESPONSE:\n" + response
        status = "audited" if result is not None else "model_error"
        anchor_reviews = result["anchor_reviews"] if result else []
        edge_reviews = result["edge_reviews"] if result else []
        missing_candidates = result["missing_anchor_candidates"] if result else []
        anchor_counts = {verdict: sum(item["verdict"] == verdict for item in anchor_reviews) for verdict in sorted(_ANCHOR_VERDICTS)}
        edge_counts = {verdict: sum(item["verdict"] == verdict for item in edge_reviews) for verdict in sorted(_EDGE_VERDICTS)}
        results.append({
            "event_id": event_id,
            "audit_status": status,
            "source_outcome_sha256": relation_record["source_outcome_sha256"],
            "source_anchor_record_sha256": canonical_sha256(anchor_record),
            "source_relation_record_sha256": canonical_sha256(relation_record),
            "relation_candidate_status": "model_error" if relation_failed else "validated_candidate",
            "relation_attempt_count": relation_record["attempt_count"] if relation_failed else relation_record["extractor"].get("attempt_count", 1),
            "relation_repair_errors": [] if relation_failed else relation_record["extractor"].get("repair_errors", []),
            "fine_edge_review_status": "not_run_source_extraction_failed" if relation_failed else "reviewed_candidates",
            "source_relation_errors": relation_record.get("errors", []) if relation_failed else [],
            "anchor_extractor_provider": anchor_record["extractor"]["provider"],
            "anchor_extractor_model": anchor_record["extractor"]["model"],
            "relation_extractor_provider": relation_record["extractor"]["provider"],
            "relation_extractor_model": relation_record["extractor"]["model"],
            "auditor_provider": provider,
            "auditor_model": model,
            "audit_prompt_version": PROMPT_VERSION,
            "anchor_count": len(anchor_record["anchors"]),
            "anchor_verdict_counts": anchor_counts,
            "anchor_reviews": anchor_reviews,
            "missing_anchor_candidates": missing_candidates,
            "edge_count": None if relation_failed else len(relation_record.get("edges", [])),
            "edge_verdict_counts": None if relation_failed else edge_counts,
            "edge_reviews": None if relation_failed else edge_reviews,
            "errors": errors,
            "auditor_response_sha256": response_hash,
        })
        print(f"[{index}/{len(relation_by_id)}] relationship-audit-case-{index:04d}: anchors={anchor_counts}, edges={edge_counts}, missing={len(missing_candidates)}", flush=True)

    all_anchor_reviews = [item for row in results for item in row["anchor_reviews"]]
    all_edge_reviews = [item for row in results for item in (row["edge_reviews"] or [])]
    all_missing = [item for row in results for item in row["missing_anchor_candidates"]]
    anchor_totals = {verdict: sum(item["verdict"] == verdict for item in all_anchor_reviews) for verdict in sorted(_ANCHOR_VERDICTS)}
    edge_totals = {verdict: sum(item["verdict"] == verdict for item in all_edge_reviews) for verdict in sorted(_EDGE_VERDICTS)}
    edge_relation = {
        edge["edge_id"]: edge["relation"]
        for record in relation_rows for edge in record.get("edges", [])
    }
    edge_by_relation = {
        relation: {
            verdict: sum(item["verdict"] == verdict and edge_relation[item["edge_id"]] == relation for item in all_edge_reviews)
            for verdict in sorted(_EDGE_VERDICTS)
        }
        for relation in sorted(FINE_RELATIONS)
    }
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-expression-anchor-and-relation-edge-model-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "events_sha256": _sha256(event_bytes),
        "outcomes_sha256": _sha256(outcome_bytes),
        "expression_anchors_sha256": _sha256(anchor_bytes),
        "relation_candidates_sha256": _sha256(relation_bytes),
        "auditor_provider": provider,
        "auditor_model": model,
        "anchor_extraction_provider": anchor_extractor["provider"],
        "anchor_extraction_model": anchor_extractor["model"],
        "relation_extraction_provider": relation_extractor["provider"],
        "relation_extraction_model": relation_extractor["model"],
        "event_count": len(results),
        "successful_event_count": sum(row["audit_status"] == "audited" for row in results),
        "relation_extraction_error_event_count": sum(row["relation_candidate_status"] == "model_error" for row in results),
        "relation_repaired_event_count": sum(bool(row["relation_repair_errors"]) for row in results),
        "fine_edge_review_event_count": sum(row["fine_edge_review_status"] == "reviewed_candidates" for row in results),
        "anchor_review_count": len(all_anchor_reviews),
        "anchor_verdict_counts": anchor_totals,
        "model_suggested_missing_anchor_count": len(all_missing),
        "edge_review_count": len(all_edge_reviews),
        "edge_verdict_counts": edge_totals,
        "edge_verdict_counts_by_relation": edge_by_relation,
        "auditor_same_provider_as_relation_extractor": relation_extractor["provider"] == provider,
        "auditor_same_model_as_anchor_extractor": anchor_extractor["provider"] == provider and anchor_extractor["model"] == model,
        "interpretation": "Same-provider cross-model model-as-judge diagnostics only; not accuracy, recall, human review, or semantic gold.",
        "limitations": [
            "The reviewer is a model from the same provider family as both extractors; correlated bias and mutable aliases remain possible.",
            "Anchor and edge verdicts are interpretations, not precision estimates; missing-anchor suggestions are not recall estimates.",
            "The audit evaluates only supplied candidate anchors/edges and a bounded sample of possible omissions; it cannot establish completeness.",
            "Only transcript-derived mention-to-idea relationships are reviewed; no pre-event news/context or inter-event graph is supplied.",
        ],
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
        "output": str(output_path), "manifest": str(manifest_path), "events": len(results),
        "anchor_reviews": len(all_anchor_reviews), "edge_reviews": len(all_edge_reviews),
        "anchor_verdict_counts": anchor_totals, "edge_verdict_counts": edge_totals,
        "missing_anchor_suggestions": len(all_missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--provider", default=AUDITOR_PROVIDER)
    parser.add_argument("--model", default=AUDITOR_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        outcomes_path=args.outcomes,
        relations_path=args.relations,
        anchors_path=args.anchors,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
