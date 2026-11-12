"""Cross-model, transcript-only audit for layered transcript outcomes.

Scores and issue flags are model-judged development diagnostics, not semantic
accuracy or ground truth. The audit deliberately does not score forecastability:
without point-in-time context, that property cannot be assessed reliably.
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

from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/outcomes/2019-layered-outcome-cross-model-audit-v2.jsonl"
AUDITOR_PROVIDER = "openai-codex"
AUDITOR_MODEL = "gpt-6.1-sol"
PROMPT_VERSION = "layered-outcome-transcript-audit-no-forecastability-score-v1"
_SYSTEM_PROMPT = (
    "You are a skeptical transcript-only auditor. "
    "Use only the supplied transcript and outcome representation. Do not use tools, outside knowledge, event identity, "
    "news, forecasts, or prior labels. Return one JSON object and no markdown."
)
_ISSUE_FLAGS = {
    "unsupported", "overstated", "understated", "conflated", "over_split", "duplicate",
    "wrong_link", "missing_direct_expression", "vague",
}


class _PermanentProviderError(RuntimeError):
    """Provider authentication/billing rejection; retrying other events is futile."""


def _is_permanent_provider_rejection(output: str) -> bool:
    lowered = output.casefold()
    return any(marker in lowered for marker in (
        "insufficient balance", "invalid api key", "unauthorized", "authentication failed", "forbidden",
    ))


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
                raise ValueError("target-speaker turn is missing a turn ID")
            turns.append({"turn_id": turn_id, "text": text.strip()})
    return turns


def _blind(turns: list[dict[str, str]], case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    blind, mapping = [], {}
    for index, turn in enumerate(turns):
        blind_id = f"{case_id}:turn:{index:04d}"
        blind.append({"turn_id": blind_id, "text": turn["text"]})
        mapping[blind_id] = turn["turn_id"]
    return blind, mapping


def _select_idea_ids(ideas: list[dict[str, Any]], max_reviews: int) -> list[str]:
    """Review all ideas up to the cap; otherwise round-robin across strata."""
    if len(ideas) <= max_reviews:
        return [idea["idea_id"] for idea in ideas]
    strata: dict[tuple[str, bool], list[str]] = {}
    for idea in ideas:
        key = (str(idea.get("prominence", "unknown")), idea.get("forecastable") is True)
        strata.setdefault(key, []).append(idea["idea_id"])
    for values in strata.values():
        values.sort()
    selected = []
    while len(selected) < max_reviews and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < max_reviews:
                selected.append(strata[key].pop(0))
    return selected


def _prompt(
    blind_turns: list[dict[str, str]],
    row: dict[str, Any],
    review_idea_ids: list[str],
) -> str:
    selected = set(review_idea_ids)
    ideas = [idea for idea in row["ideas"] if idea["idea_id"] in selected]
    task = """Audit the layered representation against the supplied complete target-speaker transcript.

Definitions: a mention is transcript-local content/evidence; an idea is an independently meaningful unit of thought; typed links state whether a mention expresses, supports, illustrates, elaborates, qualifies, or contextualizes an idea. Be skeptical about quote entailment, abstraction, split/merge decisions, and link semantics. Do not treat every name, statistic, example, or ceremony as a separate idea, but do not discard an independently meaningful event-specific subject. Do not rewrite candidate labels.

Rate these event-level dimensions independently from 0 to 4:
- mention_coverage: 4 = salient substantive transcript units are broadly represented; 3 = modest omissions; 2 = significant omissions; 1 = multiple major omissions; 0 = most substantive content is missing.
- idea_coverage: 4 = salient independently meaningful ideas are represented without obvious major omissions; 3 = one modest omission; 2 = significant omission(s); 1 = multiple major omissions; 0 = misses nearly all ideas.
- idea_grounding: 4 = ideas are well supported by their linked transcript mentions; 3 = minor inference; 2 = material weak support; 1 = major support problems; 0 = mostly unsupported.
- idea_atomicity: 4 = ideas are appropriately distinct and scoped; 3 = minor over/under-splitting; 2 = noticeable conflation or fragmentation; 1 = serious problems; 0 = unusable grouping.
- link_quality: 4 = typed relations fit; 3 = minor mismatches; 2 = several material mismatches; 1 = most links misrepresent the relation; 0 = links are unusable.

Do NOT score forecastability or claim these ideas could actually have been predicted: no pre-event context is supplied. You may note whether event-specific content was preserved separately from forecasting labels, but actual forecastability is unassessed.

For every requested idea_review_id, return one review with support and atomicity scores (0–4), zero or more issue flags, and a brief reason. Reviews are a sample only when review_idea_ids is shorter than the supplied idea list. Return at most 12 issue_examples and at most 10 missing_content examples. Each example must cite one or more exact contiguous quotes from supplied turns; do not reconstruct quotes. `target_ids` must be supplied mention/idea IDs; for a missing-content example, use an empty list.

Return only this JSON object:
{
  "mention_coverage": 0,
  "idea_coverage": 0,
  "idea_grounding": 0,
  "idea_atomicity": 0,
  "link_quality": 0,
  "idea_reviews": [
    {"idea_id": "requested ID", "support": 0, "atomicity": 0, "issue_flags": [], "reason": "short note"}
  ],
  "issue_examples": [
    {"target_ids": ["supplied IDs"], "issue_flag": "unsupported", "note": "short transcript-grounded reason", "evidence": [{"turn_id": "blind turn ID", "exact_quote": "exact substring"}]}
  ],
  "missing_content": [
    {"description": "short transcript-grounded omission", "evidence": [{"turn_id": "blind turn ID", "exact_quote": "exact substring"}]}
  ],
  "overall_note": "brief note; state forecastability was not assessed without pre-event context"
}

Allowed issue flags: unsupported, overstated, understated, conflated, over_split, duplicate, wrong_link, missing_direct_expression, vague.

AUDIT INPUT:
"""
    outcome = {
        "mentions": row["mentions"],
        "ideas": row["ideas"],
        "links": row["links"],
        "idea_review_ids": review_idea_ids,
    }
    return task + json.dumps({"turns": blind_turns, "layered_outcome": outcome}, ensure_ascii=False)


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


def _validate(
    value: Any,
    *,
    review_ids: set[str],
    mention_ids: set[str],
    idea_ids: set[str],
    blind_turns: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("audit response must be an object")
    scores = {}
    for field in ("mention_coverage", "idea_coverage", "idea_grounding", "idea_atomicity", "link_quality"):
        score = value.get(field)
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
            raise ValueError(f"{field} must be an integer from 0 to 4")
        scores[field] = score
    reviews = value.get("idea_reviews")
    if not isinstance(reviews, list):
        raise ValueError("idea_reviews must be a list")
    by_id = {}
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("each idea review must be an object")
        idea_id = review.get("idea_id")
        if not isinstance(idea_id, str) or idea_id not in review_ids or idea_id in by_id:
            raise ValueError("idea review IDs must exactly match the requested review sample")
        item = {"idea_id": idea_id}
        for field in ("support", "atomicity"):
            score = review.get(field)
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                raise ValueError(f"{idea_id}.{field} must be an integer from 0 to 4")
            item[field] = score
        flags = review.get("issue_flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) or flag not in _ISSUE_FLAGS for flag in flags):
            raise ValueError(f"{idea_id}.issue_flags contains an unsupported value")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{idea_id}.reason must be nonempty")
        item.update({"issue_flags": flags, "reason": reason.strip()})
        by_id[idea_id] = item
    if set(by_id) != review_ids:
        raise ValueError("auditor omitted one or more requested idea IDs")

    issue_examples = value.get("issue_examples")
    if not isinstance(issue_examples, list) or len(issue_examples) > 12:
        raise ValueError("issue_examples must be a list of at most 12 objects")
    checked_issues = []
    for example in issue_examples:
        if not isinstance(example, dict) or not isinstance(example.get("issue_flag"), str) or example.get("issue_flag") not in _ISSUE_FLAGS:
            raise ValueError("issue example has an unsupported issue flag")
        targets = example.get("target_ids")
        if not isinstance(targets, list) or any(not isinstance(target, str) or target not in mention_ids | idea_ids for target in targets):
            raise ValueError("issue example references an unknown target ID")
        note = example.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("issue example note must be nonempty")
        evidence = _validate_evidence(example.get("evidence"), blind_turns)
        checked_issues.append({"target_ids": targets, "issue_flag": example["issue_flag"], "note": note.strip(), "evidence": evidence})

    missing = value.get("missing_content")
    if not isinstance(missing, list) or len(missing) > 10:
        raise ValueError("missing_content must be a list of at most 10 objects")
    checked_missing = []
    for item in missing:
        if not isinstance(item, dict) or not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ValueError("each missing-content example needs a description")
        checked_missing.append({"description": item["description"].strip(), "evidence": _validate_evidence(item.get("evidence"), blind_turns)})
    note = value.get("overall_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("overall_note must be nonempty")
    return {
        **scores,
        "idea_reviews": [by_id[item] for item in sorted(by_id)],
        "issue_examples": checked_issues,
        "missing_content": checked_missing,
        "overall_note": note.strip(),
    }


def _validate_evidence(value: Any, blind_turns: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("each audit example needs at least one exact transcript quote")
    evidence = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("audit evidence must be an object")
        turn_id, quote = item.get("turn_id"), item.get("exact_quote")
        if not isinstance(turn_id, str) or turn_id not in blind_turns or not isinstance(quote, str) or quote not in blind_turns[turn_id]:
            raise ValueError("audit evidence quote is not an exact substring of its cited blind turn")
        evidence.append({"turn_id": turn_id, "exact_quote": quote})
    return evidence


def _call_pi(prompt: str, *, provider: str, model: str, timeout: int) -> str:
    executable = shutil.which("pi")
    if not executable:
        raise RuntimeError("pi CLI is required for the audit")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".audit-prompt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        result = subprocess.run([
            executable, "--provider", provider, "--model", model,
            "--print", "--mode", "text", "--no-tools", "--no-extensions", "--no-mcp",
            "--no-skills", "--no-context-files", "--no-session", "--system-prompt", _SYSTEM_PROMPT,
            f"@{prompt_path}",
        ], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            provider_output = result.stdout + "\n" + result.stderr
            if _is_permanent_provider_rejection(provider_output):
                raise _PermanentProviderError("pi request rejected for provider authentication or account balance")
            raise RuntimeError(f"pi exited with status {result.returncode}")
        if not result.stdout.strip():
            raise RuntimeError("pi returned an empty response")
        return result.stdout.strip()
    finally:
        prompt_path.unlink(missing_ok=True)


def _select_sample(
    rows: list[dict[str, Any]],
    *,
    events: dict[str, dict[str, Any]],
    auditor_provider: str,
    auditor_model: str,
    sample_size: int,
    include_event_ids: set[str],
    allow_same_provider: bool = False,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    by_id = {row.get("event_id"): row for row in rows}
    missing = include_event_ids - set(by_id)
    if missing:
        raise ValueError(f"explicit audit events are missing from outcome artifacts: {sorted(missing)}")
    for event_id in include_event_ids:
        extractor = by_id[event_id].get("extractor", {})
        if extractor.get("provider") == auditor_provider and extractor.get("model") == auditor_model:
            raise ValueError(f"explicit event {event_id} uses the exact audit model for extraction")
        if extractor.get("provider") == auditor_provider and not allow_same_provider:
            raise ValueError(f"explicit event {event_id} uses the audit provider; pass allow_same_provider for a correlated cross-model diagnostic")
    eligible = []
    excluded_same_provider, excluded_same_model = [], []
    for row in rows:
        event_id = row.get("event_id")
        extractor = row.get("extractor", {})
        extraction_provider, extraction_model = extractor.get("provider"), extractor.get("model")
        if extraction_provider == auditor_provider and extraction_model == auditor_model:
            excluded_same_model.append(event_id)
            continue
        if extraction_provider == auditor_provider and not allow_same_provider and event_id not in include_event_ids:
            excluded_same_provider.append(event_id)
            continue
        if event_id not in include_event_ids and row.get("status") == "validated_candidate" and event_id in events:
            eligible.append(row)
    eligible.sort(key=lambda row: (
        row.get("target_speaker_word_count", 0),
        row.get("event_id", ""),
    ))
    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    if sample_size and not eligible:
        raise ValueError("no eligible cross-model outcomes are available for the requested sample")
    if sample_size > len(eligible):
        sample_size = len(eligible)
    sampled = []
    if sample_size == 1:
        sampled = [eligible[len(eligible) // 2]]
    elif sample_size > 1:
        indices = [round(index * (len(eligible) - 1) / (sample_size - 1)) for index in range(sample_size)]
        sampled = [eligible[index] for index in indices]
    selected_by_id = {row["event_id"]: row for row in sampled}
    for event_id in sorted(include_event_ids):
        selected_by_id[event_id] = by_id[event_id]
    return list(selected_by_id.values()), sorted(excluded_same_provider), sorted(excluded_same_model)


def run(
    *,
    events_path: Path,
    outcome_paths: list[Path],
    output_path: Path,
    provider: str = AUDITOR_PROVIDER,
    model: str = AUDITOR_MODEL,
    timeout: int = 300,
    sample_size: int = 12,
    include_event_ids: set[str] | None = None,
    max_idea_reviews: int = 80,
    allow_same_provider: bool = False,
) -> dict[str, Any]:
    events, event_bytes = _read_jsonl(events_path)
    event_by_id = {event.get("event_id"): event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("event corpus contains duplicate event IDs")
    outcome_rows, artifact_hashes = [], []
    for path in outcome_paths:
        rows, raw = _read_jsonl(path)
        artifact_hashes.append({"path": path.name, "sha256": _sha256(raw)})
        outcome_rows.extend(rows)
    outcome_by_id = {}
    for row in outcome_rows:
        event_id = row.get("event_id")
        if event_id in outcome_by_id:
            raise ValueError(f"duplicate outcome event ID across artifacts: {event_id}")
        outcome_by_id[event_id] = row
    if max_idea_reviews < 1:
        raise ValueError("max_idea_reviews must be at least 1")
    selected, excluded_same_provider, excluded_same_model = _select_sample(
        outcome_rows,
        events=event_by_id,
        auditor_provider=provider,
        auditor_model=model,
        sample_size=sample_size,
        include_event_ids=include_event_ids or set(),
        allow_same_provider=allow_same_provider,
    )
    selected.sort(key=lambda row: (row.get("event_date", ""), row.get("event_id", "")))
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite layered outcome audit: {output_path}")

    audited = []
    prompt_template_hash = _sha256((_SYSTEM_PROMPT + "\n" + _prompt([], {"mentions": [], "ideas": [], "links": []}, [])).encode("utf-8"))
    for index, row in enumerate(selected, 1):
        event_id = row["event_id"]
        event = event_by_id[event_id]
        if row.get("status") != "validated_candidate":
            raise ValueError(f"cannot audit nonvalidated outcome row: {event_id}")
        if row.get("source_record_sha256") != _canonical_hash(event):
            raise ValueError(f"source-record hash mismatch for {event_id}")
        turns = _target_turns(event)
        transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
        if row.get("target_transcript_sha256") != _sha256(transcript.encode("utf-8")):
            raise ValueError(f"transcript hash mismatch for {event_id}")
        source_turns = {turn["turn_id"]: turn["text"] for turn in turns}
        validate_layered_outcome(row, source_turns)
        blind, mapping = _blind(turns, f"audit-case-{index:04d}")
        source_to_blind = {source_id: blind_id for blind_id, source_id in mapping.items()}
        review_ids = _select_idea_ids(row["ideas"], max_idea_reviews)
        blind_mentions = []
        for mention in row["mentions"]:
            blind_mentions.append({
                **{key: value for key, value in mention.items() if key != "evidence"},
                "evidence": [
                    {"turn_id": source_to_blind[item["turn_id"]], "exact_quote": item["exact_quote"]}
                    for item in mention["evidence"]
                ],
            })
        audit_row = {
            **{key: value for key, value in row.items() if key not in {"mentions", "ideas", "links", "validation", "extractor"}},
            "mentions": blind_mentions,
            "ideas": row["ideas"],
            "links": row["links"],
        }
        prompt = _prompt(blind, audit_row, review_ids)
        errors, result, response_hash = [], None, None
        for attempt in range(3):
            response = ""
            try:
                response = _call_pi(prompt, provider=provider, model=model, timeout=timeout)
                response_hash = _sha256(response.encode("utf-8"))
                blind_turns = {turn["turn_id"]: turn["text"] for turn in blind}
                result = _validate(
                    _parse_object(response),
                    review_ids=set(review_ids),
                    mention_ids={mention["mention_id"] for mention in row["mentions"]},
                    idea_ids={idea["idea_id"] for idea in row["ideas"]},
                    blind_turns=blind_turns,
                )
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, _PermanentProviderError):
                    raise
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < 2:
                    prompt += (
                        "\n\nFORMAT REPAIR: Fix the previous response to satisfy the required scores, IDs, and exact quote citations. "
                        "Do not change the outcome or use outside sources. Return the corrected JSON only.\nPREVIOUS RESPONSE:\n" + response
                    )
        if result is None:
            audited.append({
                "event_id": event_id,
                "event_date": event.get("event_date"),
                "audit_status": "model_error",
                "errors": errors,
                "extraction_provider": row.get("extractor", {}).get("provider"),
                "extraction_model": row.get("extractor", {}).get("model"),
                "auditor_provider": provider,
                "auditor_model": model,
            })
            print(f"[{index}/{len(selected)}] audit-case-{index:04d}: model_error", flush=True)
            continue
        def remap_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {**item, "evidence": [
                    {"turn_id": mapping[ev["turn_id"]], "exact_quote": ev["exact_quote"]}
                    for ev in item["evidence"]
                ]}
                for item in items
            ]
        audited.append({
            "event_id": event_id,
            "event_date": event.get("event_date"),
            "source_record_sha256": row["source_record_sha256"],
            "target_transcript_sha256": row["target_transcript_sha256"],
            "extraction_provider": row.get("extractor", {}).get("provider"),
            "extraction_model": row.get("extractor", {}).get("model"),
            "auditor_provider": provider,
            "auditor_model": model,
            "audit_prompt_version": PROMPT_VERSION,
            "audit_status": "audited",
            "idea_count": len(row["ideas"]),
            "idea_review_count": len(result["idea_reviews"]),
            "idea_review_coverage_percent": round(100 * len(result["idea_reviews"]) / len(row["ideas"]), 2) if row["ideas"] else 100.0,
            "dimensions_0_to_4": {key: result[key] for key in ("mention_coverage", "idea_coverage", "idea_grounding", "idea_atomicity", "link_quality")},
            "forecastability_assessment": "not_assessed_without_pre_event_context",
            "idea_reviews": result["idea_reviews"],
            "issue_examples": remap_evidence(result["issue_examples"]),
            "missing_content": remap_evidence(result["missing_content"]),
            "overall_note": result["overall_note"],
            "auditor_response_sha256": response_hash,
            "auditor_attempts": len(errors) + 1,
            "auditor_errors": errors,
        })
        print(f"[{index}/{len(selected)}] audit-case-{index:04d}: audited (ideas={len(row['ideas'])}, reviewed={len(result['idea_reviews'])})", flush=True)

    successful = [row for row in audited if row["audit_status"] == "audited"]
    dimensions = ("mention_coverage", "idea_coverage", "idea_grounding", "idea_atomicity", "link_quality")
    means = {
        dimension: sum(row["dimensions_0_to_4"][dimension] for row in successful) / len(successful) if successful else None
        for dimension in dimensions
    }
    per_event_idea_reviews = [review for row in successful for review in row["idea_reviews"]]
    flag_counts: dict[str, int] = {}
    for review in per_event_idea_reviews:
        for flag in review["issue_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audited).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "layered-transcript-cross-model-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": prompt_template_hash,
        "events_sha256": _sha256(event_bytes),
        "outcome_artifacts": artifact_hashes,
        "auditor_provider": provider,
        "auditor_model": model,
        "sample_strategy": "evenly spaced by transcript word-count rank among validated eligible candidates, plus explicit include-event-id values",
        "same_provider_cross_model_allowed": allow_same_provider,
        "sampled_event_ids": [row.get("event_id") for row in audited],
        "excluded_same_provider_event_ids": excluded_same_provider,
        "excluded_same_model_event_ids": excluded_same_model,
        "event_count": len(audited),
        "successful_audit_count": len(successful),
        "audit_error_count": len(audited) - len(successful),
        "event_macro_means_0_to_4": means,
        "idea_review_count": len(per_event_idea_reviews),
        "idea_issue_flag_counts": flag_counts,
        "forecastability_scored": False,
        "score_interpretation": "Cross-model AI audit diagnostic only; ratings are not accuracy, human review, or semantic gold.",
        "same_provider_event_ids": [row["event_id"] for row in successful if row.get("extraction_provider") == provider],
        "cross_provider_event_ids": [row["event_id"] for row in successful if row.get("extraction_provider") != provider],
        "limitations": [
            "All sampled outcomes are from an already-inspected development slice or a single long-speech smoke test; no independent test performance is measured.",
            "The auditor is another model; correlated error, provider bias, mutable aliases, and overconfidence remain possible.",
            "Same-provider cross-model cases may share training data, alignment, and systematic errors with the extractor.",
            "Completeness, atomicity, support, and link ratings are subjective judgments without a trusted reference inventory.",
            "Forecastability is not scored because no pre-event context is supplied to the auditor.",
            "Outcome extraction and audit both use the target transcript; this is only a label-quality audit, never a prediction-input evaluation.",
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
    return {"output": str(output_path), "manifest": str(manifest_path), "events": len(successful), "errors": len(audited) - len(successful), "dimension_means_0_to_4": means}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--outcome-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--provider", default=AUDITOR_PROVIDER)
    parser.add_argument("--model", default=AUDITOR_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--include-event-id", action="append", default=[])
    parser.add_argument("--max-idea-reviews", type=int, default=80)
    parser.add_argument("--allow-same-provider", action="store_true", help="allow explicitly labeled same-provider, different-model diagnostics")
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        outcome_paths=args.outcome_artifact,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
        sample_size=args.sample_size,
        include_event_ids=set(args.include_event_id),
        max_idea_reviews=args.max_idea_reviews,
        allow_same_provider=args.allow_same_provider,
    ), indent=2))


if __name__ == "__main__":
    main()
