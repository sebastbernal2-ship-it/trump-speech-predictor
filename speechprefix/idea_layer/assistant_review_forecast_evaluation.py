"""Score frozen forecast candidates against assistant-reviewed reference decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .assistant_review_validation import _read_jsonl

_ALLOWED = {"match", "partial_match", "not_supported", "uncertain"}


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_report(path: str | Path) -> tuple[dict[str, Any], str]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("forecast report must be a JSON object")
    expected = report.get("report_content_sha256")
    payload = dict(report)
    payload.pop("report_content_sha256", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError("forecast report content hash does not validate")
    return report, _file_sha256(path)


def evaluate_assistant_reviewed_forecasts(
    *,
    forecast_path: str | Path,
    reference_paths: Iterable[str | Path],
    decisions_path: str | Path,
    protocol_snapshot_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate assistant decisions; never infer semantic matches automatically.

    Each decision row must cite mapped transcript turns for supported/partial/
    uncertain decisions. ``not_supported`` rows must have no matched outcome IDs.
    Incomplete/empty references are excluded rather than counted as misses.
    """
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")
    forecast, forecast_sha = _verified_report(forecast_path)
    protocol_sha = _file_sha256(protocol_snapshot_path)
    forecast_rows = forecast.get("forecasts")
    if not isinstance(forecast_rows, list):
        raise ValueError("forecast report must contain forecasts")
    forecasts: dict[str, dict[str, Any]] = {}
    for row in forecast_rows:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in forecasts:
            raise ValueError("forecast event IDs must be unique nonempty strings")
        forecasts[event_id] = row

    references: dict[str, tuple[dict[str, Any], str]] = {}
    reference_hashes: dict[str, str] = {}
    for value in reference_paths:
        path = Path(value)
        rows, file_hash = _read_jsonl(path)
        reference_hashes[str(path)] = file_hash
        for row in rows:
            event_id = row.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in references:
                raise ValueError(f"duplicate or invalid reference event_id: {event_id!r}")
            if row.get("review_status") not in {
                "assistant-reviewed-transcript-only",
                "assistant-reviewed-transcript-only-posthoc-mapped",
            }:
                # Existing v1 mapped development rows use the first value; leave
                # other review types (notably the contaminated 2026 audit) out.
                if not str(row.get("review_status", "")).startswith("assistant-reviewed-transcript-only"):
                    raise ValueError(f"{event_id}: reference is not transcript-only assistant review")
            references[event_id] = (row, file_hash)

    decision_rows, decisions_file_hash = _read_jsonl(decisions_path)
    decisions_by_event: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in decisions_by_event:
            raise ValueError(f"duplicate or invalid decision event_id: {event_id!r}")
        decisions_by_event[event_id] = row

    if set(forecasts) != set(references):
        raise ValueError("forecast and assistant-reference event sets differ")
    expected_scored = {
        event_id for event_id, (reference, _) in references.items()
        if any(idea.get("forecastable") is True for idea in reference.get("outcome_ideas", []))
    }
    if set(decisions_by_event) != expected_scored:
        raise ValueError(
            "decision event set must include exactly references with forecastable outcomes; "
            f"missing={sorted(expected_scored-set(decisions_by_event))}, "
            f"extra={sorted(set(decisions_by_event)-expected_scored)}"
        )

    scored_events: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in ("match", "partial_match", "not_supported", "uncertain")}
    outcome_ids = {"strict": set(), "strict_or_partial": set(), "including_uncertain": set()}
    all_forecastable_ids: set[str] = set()
    for event_id in sorted(expected_scored, key=lambda value: (
        forecasts[value]["event_date"], value,
    )):
        reference, _ = references[event_id]
        forecast_row = forecasts[event_id]
        decision_row = decisions_by_event[event_id]
        if decision_row.get("forecast_report_sha256") != forecast_sha:
            raise ValueError(f"{event_id}: decision row references a different forecast report")
        if decision_row.get("target_transcript_sha256") != reference.get("target_transcript_sha256"):
            raise ValueError(f"{event_id}: decision row target transcript hash differs from reference")
        if decision_row.get("source_record_sha256") != reference.get("source_record_sha256"):
            raise ValueError(f"{event_id}: decision row source-record hash differs from reference")
        if decision_row.get("event_date") != reference.get("event_date") or decision_row.get("event_date") != forecast_row.get("event_date"):
            raise ValueError(f"{event_id}: event date mismatch")
        if decision_row.get("review_status") != "assistant-reviewed-posthoc-nonblind":
            raise ValueError(f"{event_id}: post-hoc matching status must be explicit")
        reviewer = decision_row.get("reviewer")
        if not isinstance(reviewer, dict) or reviewer.get("reviewer_type") != "AI assistant":
            raise ValueError(f"{event_id}: missing AI reviewer provenance")
        if reviewer.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"{event_id}: reviewer protocol hash does not match supplied immutable protocol snapshot")

        target_ideas = {
            idea["idea_id"]: idea for idea in reference.get("outcome_ideas", [])
            if idea.get("forecastable") is True
        }
        all_forecastable_ids.update(f"{event_id}:{idea_id}" for idea_id in target_ideas)
        candidate_by_id = {candidate["idea_id"]: candidate for candidate in forecast_row.get("predictions", [])}
        if len(candidate_by_id) != len(forecast_row.get("predictions", [])):
            raise ValueError(f"{event_id}: duplicate candidate IDs in forecast")
        decisions = decision_row.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError(f"{event_id}: decisions must be a list")
        decision_by_id: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            candidate_id = decision.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in decision_by_id:
                raise ValueError(f"{event_id}: duplicate or invalid candidate decision")
            decision_by_id[candidate_id] = decision
        if set(decision_by_id) != set(candidate_by_id):
            raise ValueError(f"{event_id}: decisions do not cover exactly the forecast candidates")

        event_decisions: list[dict[str, Any]] = []
        for candidate_id, candidate in candidate_by_id.items():
            decision = decision_by_id[candidate_id]
            if decision.get("rank") != candidate.get("rank") or decision.get("candidate_idea") != candidate.get("idea"):
                raise ValueError(f"{event_id}/{candidate_id}: decision is not bound to forecast rank and surface")
            label = decision.get("decision")
            if label not in _ALLOWED:
                raise ValueError(f"{event_id}/{candidate_id}: unsupported decision {label!r}")
            matched = decision.get("matched_outcome_idea_ids")
            citations = decision.get("evidence_turn_ids")
            rationale = decision.get("rationale")
            if not isinstance(matched, list) or len(set(matched)) != len(matched):
                raise ValueError(f"{event_id}/{candidate_id}: matched_outcome_idea_ids must be unique list")
            if not isinstance(citations, list) or len(set(citations)) != len(citations):
                raise ValueError(f"{event_id}/{candidate_id}: evidence_turn_ids must be unique list")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"{event_id}/{candidate_id}: rationale is required")
            if not citations:
                raise ValueError(f"{event_id}/{candidate_id}: every decision needs transcript-turn citations")
            all_reference_turns = {
                turn_id for idea in reference.get("outcome_ideas", [])
                for turn_id in idea.get("evidence_turn_ids", [])
            }
            if not set(citations) <= all_reference_turns:
                raise ValueError(f"{event_id}/{candidate_id}: citations are not present in the reviewed transcript outcomes")
            if label == "not_supported":
                if matched:
                    raise ValueError(f"{event_id}/{candidate_id}: not_supported may not claim matched outcome IDs")
            else:
                if not matched or not set(matched) <= set(target_ideas):
                    raise ValueError(f"{event_id}/{candidate_id}: supported decision needs forecastable outcomes")
                cited_by_matches = {
                    turn_id for idea_id in matched
                    for turn_id in target_ideas[idea_id].get("evidence_turn_ids", [])
                }
                if not set(citations) <= cited_by_matches:
                    raise ValueError(f"{event_id}/{candidate_id}: cited turns do not support the matched outcome IDs")
                if label == "match":
                    outcome_ids["strict"].update(f"{event_id}:{idea_id}" for idea_id in matched)
                if label in {"match", "partial_match"}:
                    outcome_ids["strict_or_partial"].update(f"{event_id}:{idea_id}" for idea_id in matched)
                outcome_ids["including_uncertain"].update(f"{event_id}:{idea_id}" for idea_id in matched)
            aggregate[label] += 1
            event_decisions.append({
                "rank": candidate["rank"],
                "candidate_id": candidate_id,
                "candidate_idea": candidate["idea"],
                "decision": label,
                "matched_outcome_idea_ids": matched,
                "evidence_turn_ids": citations,
                "rationale": rationale,
            })
        missed = sorted(set(target_ideas) - {
            idea_id.split(":", 1)[1] for idea_id in outcome_ids["strict_or_partial"]
            if idea_id.startswith(event_id + ":")
        })
        event_decisions.sort(key=lambda item: item["rank"])
        scored_events.append({
            "event_id": event_id,
            "event_date": reference["event_date"],
            "forecastable_outcome_count": len(target_ideas),
            "candidate_count": len(event_decisions),
            "candidate_decisions": event_decisions,
            "strict_match_candidate_count": sum(item["decision"] == "match" for item in event_decisions),
            "partial_candidate_count": sum(item["decision"] == "partial_match" for item in event_decisions),
            "uncertain_candidate_count": sum(item["decision"] == "uncertain" for item in event_decisions),
            "unsupported_candidate_count": sum(item["decision"] == "not_supported" for item in event_decisions),
            "outcome_idea_ids_missed_strictly": sorted(set(target_ideas) - {
                idea_id.split(":", 1)[1] for idea_id in outcome_ids["strict"]
                if idea_id.startswith(event_id + ":")
            }),
            "outcome_idea_ids_without_match_or_partial": missed,
            "coverage_assessment": reference.get("coverage_assessment"),
        })

    candidate_count = sum(row["candidate_count"] for row in scored_events)
    outcome_count = len(all_forecastable_ids)
    is_news_forecast = str(forecast.get("artifact_type", "")).startswith("retrospective-news-")
    is_semantic_forecast = forecast.get("artifact_type") == "semantic-idea-forecast-evaluation-bundle"
    if is_news_forecast:
        evaluation_mode = "retrospective news-to-idea phrase forecast compared post-hoc with assistant-reviewed references"
        model_limitation = "News candidates are compact-lexicon phrases ranked by retrospective search-index coverage; retrieval after the event does not establish pre-event availability."
    elif is_semantic_forecast:
        evaluation_mode = "semantic event-conditioned idea forecast compared post-hoc with assistant-reviewed references"
        model_limitation = "Semantic proposals, model-assessed ratings, source-quality priors, and ranking scores are uncalibrated; source citations do not prove entailment or point-in-time availability."
    else:
        evaluation_mode = "retrospective history-only ranking compared post-hoc with assistant-reviewed references"
        model_limitation = "The fixed history baseline ranks common lexical phrase labels and does not use point-in-time news or event details."
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "assistant-reviewed-forecast-agreement-evaluation",
        "review_status": "assistant-reviewed-posthoc-nonblind",
        "evaluation_mode": evaluation_mode,
        "forecast_model_family": forecast.get("model_family"),
        "forecast_path": str(forecast_path),
        "forecast_report_sha256": forecast_sha,
        "review_protocol_snapshot_path": str(protocol_snapshot_path),
        "review_protocol_sha256": protocol_sha,
        "reference_file_sha256": reference_hashes,
        "decision_file_sha256": decisions_file_hash,
        "scored_event_count": len(scored_events),
        "excluded_event_count_no_forecastable_outcomes": len(forecasts) - len(scored_events),
        "candidate_decision_count": candidate_count,
        "candidate_decision_counts": aggregate,
        "candidate_agreement": {
            "strict_match_rate": aggregate["match"] / candidate_count if candidate_count else 0.0,
            "strict_plus_partial_rate": (aggregate["match"] + aggregate["partial_match"]) / candidate_count if candidate_count else 0.0,
            "uncertain_rate": aggregate["uncertain"] / candidate_count if candidate_count else 0.0,
            "unsupported_rate": aggregate["not_supported"] / candidate_count if candidate_count else 0.0,
        },
        "forecastable_outcome_count": outcome_count,
        "outcome_coverage": {
            "strict_matched_count": len(outcome_ids["strict"]),
            "strict_match_rate": len(outcome_ids["strict"]) / outcome_count if outcome_count else 0.0,
            "strict_or_partial_count": len(outcome_ids["strict_or_partial"]),
            "strict_or_partial_rate": len(outcome_ids["strict_or_partial"]) / outcome_count if outcome_count else 0.0,
            "including_uncertain_count": len(outcome_ids["including_uncertain"]),
            "including_uncertain_rate": len(outcome_ids["including_uncertain"]) / outcome_count if outcome_count else 0.0,
            "unmatched_count": outcome_count - len(outcome_ids["strict_or_partial"]),
        },
        "events": scored_events,
        "limitations": [
            "All candidate match decisions are AI judgments against assistant-reviewed references, not objective semantic accuracy or independent validation.",
            "This is post-hoc development analysis; no model selection or tuning should use these results as a clean test.",
            model_limitation,
            "Historical transcript availability is not independently verified; chronological event dates alone do not establish real-time availability.",
            "Outcome coverage counts only distinct assistant-reviewed forecastable ideas; low-completeness or no-label events are excluded, not treated as empty ground truth.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite evaluation report: {path}") from None
    return report
