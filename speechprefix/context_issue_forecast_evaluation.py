"""After-event scoring for an immutable issue forecast made before its cutoff."""

from __future__ import annotations
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_snapshots import validate_prospective_snapshot
from .context_theme_ranker import read_jsonl
from .headline_idea_matching import _target_turns, extract_transcript_ideas


def _precision_recall(predicted: list[str], target: set[str], k: int) -> dict[str, float]:
    hits = len(set(predicted[:k]) & target)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(target) if target else 0.0,
    }


def evaluate_prospective_issue_forecast(
    forecast_path: str | Path,
    snapshot_path: str | Path,
    event_records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    top_k: tuple[int, ...] = (4, 10, 25),
) -> dict[str, Any]:
    """Score saved pre-cutoff candidate ranks against a transcript and history baseline."""
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k must contain positive values")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")
    try:
        forecast = json.loads(Path(forecast_path).read_text(encoding="utf-8"))
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid forecast/snapshot JSON: {error}") from None
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be a JSON object")
    if forecast.get("forecast_type") != "prospective-context-issue-phrase-candidates":
        raise ValueError("input is not a prospective issue-phrase forecast")
    snapshot_validation = validate_prospective_snapshot(snapshot)
    if not isinstance(snapshot.get("snapshot_sha256"), str):
        raise ValueError("evaluation requires a sealed prospective snapshot")
    forecast_digest = forecast.get("report_content_sha256")
    if not isinstance(forecast_digest, str):
        raise ValueError("forecast report is missing its content hash")
    forecast_payload = {key: value for key, value in forecast.items() if key != "report_content_sha256"}
    canonical_forecast = json.dumps(forecast_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical_forecast.encode("utf-8")).hexdigest() != forecast_digest:
        raise ValueError("forecast report content hash does not match")

    snapshot_bytes = Path(snapshot_path).read_bytes()
    if forecast.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
        raise ValueError("forecast refers to a different prospective snapshot")
    if forecast.get("input_sha256") != hashlib.sha256(snapshot_bytes).hexdigest():
        raise ValueError("forecast input hash does not match snapshot file")
    event_id = snapshot_validation["event_id"]
    if forecast.get("event_id") != event_id:
        raise ValueError("forecast event ID does not match snapshot")
    cutoff_text = snapshot["prediction_cutoff_at"]
    cutoff = datetime.fromisoformat(cutoff_text.replace("Z", "+00:00"))
    frozen = datetime.fromisoformat(snapshot["snapshot_frozen_at"].replace("Z", "+00:00"))
    generated = datetime.fromisoformat(str(forecast.get("forecast_generated_at", "")).replace("Z", "+00:00"))
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise ValueError("forecast_generated_at must include a timezone")
    if generated < frozen or generated >= cutoff:
        raise ValueError("forecast was not generated after freeze and before cutoff")
    if forecast.get("prediction_cutoff_at") != cutoff_text:
        raise ValueError("forecast cutoff does not match snapshot")

    records = read_jsonl(event_records_path)
    records_by_id = {row.get("event_id"): row for row in records}
    if len(records_by_id) != len(records):
        raise ValueError("duplicate event IDs in transcript records")
    target_record = records_by_id.get(event_id)
    if target_record is None:
        raise ValueError(f"no transcript outcome for forecast event {event_id}")
    event_start = datetime.fromisoformat(snapshot["event_start_at"].replace("Z", "+00:00"))
    if event_start >= datetime.now(event_start.tzinfo):
        raise ValueError("cannot evaluate a forecast before its scheduled event has started")
    event_date = event_start.date()
    if target_record.get("event_date") != event_date.isoformat():
        raise ValueError("transcript event date does not match the prospective snapshot")
    turns = _target_turns(target_record)
    if not turns:
        raise ValueError("outcome record has no target-speaker turns")
    target_ideas = extract_transcript_ideas(turns)
    target_ids = {idea["idea_id"] for idea in target_ideas}
    if not target_ids:
        raise ValueError("outcome transcript yielded no issue phrases under the current lexicon")
    target_turn_ids = {idea["idea_id"]: idea["supporting_turn_ids"] for idea in target_ideas}

    history_counts: dict[str, int] = {}
    history_surfaces: dict[str, dict[str, int]] = {}
    history_event_ids = []
    for record in records:
        history_id = record.get("event_id")
        if history_id == event_id:
            continue
        raw_date = record.get("event_date")
        try:
            history_date = date.fromisoformat(raw_date)
        except (TypeError, ValueError):
            continue
        # Exclude same-day events because their exact order/availability is unknown.
        if history_date >= cutoff.date():
            continue
        historical_turns = _target_turns(record)
        if not historical_turns:
            continue
        history_ideas = extract_transcript_ideas(historical_turns)
        if not history_ideas:
            continue
        history_event_ids.append(history_id)
        for idea in history_ideas:
            idea_id = idea["idea_id"]
            history_counts[idea_id] = history_counts.get(idea_id, 0) + 1
            surfaces = history_surfaces.setdefault(idea_id, {})
            surface = idea["candidate_idea"]
            surfaces[surface] = surfaces.get(surface, 0) + idea["occurrence_count"]
    history_ranked = sorted(history_counts, key=lambda item: (-history_counts[item], item))

    candidates = forecast.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("forecast candidates must be a list")
    if forecast.get("candidate_count") != len(candidates):
        raise ValueError("forecast candidate_count does not match its candidate list")
    context_ranked = []
    candidate_rows = []
    for rank, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("idea_id"), str):
            raise ValueError("forecast candidate is missing idea_id")
        if candidate.get("rank") != rank:
            raise ValueError("forecast candidate ranks are not contiguous and ordered")
        idea_id = candidate["idea_id"]
        if idea_id in context_ranked:
            raise ValueError("forecast contains duplicate idea IDs")
        context_ranked.append(idea_id)
        candidate_rows.append({
            "rank": rank,
            "idea": candidate.get("idea"),
            "idea_id": idea_id,
            "matched_target": idea_id in target_ids,
            "supporting_turn_ids": target_turn_ids.get(idea_id, []),
            "supporting_sources": candidate.get("supporting_sources", []),
        })

    metrics = {}
    for k in top_k:
        metrics[f"top_{k}"] = {
            "saved_context_forecast": _precision_recall(context_ranked, target_ids, k),
            "speaker_history_prevalence": _precision_recall(history_ranked, target_ids, k),
        }
    report = {
        "schema_version": 1,
        "evaluation_type": "prospective-issue-forecast-after-event-lexical-overlap",
        "event_id": event_id,
        "event_date": event_date.isoformat(),
        "prediction_cutoff_at": cutoff_text,
        "forecast_generated_at": forecast["forecast_generated_at"],
        "forecast_content_sha256": forecast_digest,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "input_sha256": {
            "forecast": hashlib.sha256(Path(forecast_path).read_bytes()).hexdigest(),
            "snapshot": hashlib.sha256(snapshot_bytes).hexdigest(),
            "event_records": hashlib.sha256(Path(event_records_path).read_bytes()).hexdigest(),
        },
        "target_policy": "automatic compact issue phrases from target-speaker turns; not semantic gold",
        "history_policy": "distinct earlier archive events by date; same-day events excluded; archive availability timestamps are unverified",
        "config": {"top_k": list(top_k)},
        "history_event_count": len(history_event_ids),
        "history_event_ids": sorted(history_event_ids),
        "target_issue_phrase_count": len(target_ids),
        "metrics": metrics,
        "saved_context_candidates": candidate_rows,
        "history_prevalence_top_25": [
            {
                "rank": rank,
                "idea": max(history_surfaces[idea_id].items(), key=lambda item: (item[1], item[0]))[0],
                "idea_id": idea_id,
                "training_event_count": history_counts[idea_id],
                "matched_target": idea_id in target_ids,
                "supporting_turn_ids": target_turn_ids.get(idea_id, []),
            }
            for rank, idea_id in enumerate(history_ranked[:25], 1)
        ],
        "limitations": [
            "One event is a case result, not an estimate of generalization or predictive lift.",
            "Transcript phrase extraction is automatic and lexical; phrase overlap is not semantic equivalence.",
            "Google News RSS publication timestamps are feed-reported; event-schedule availability attestations require independent review.",
            "Historical transcript archive observation times are unavailable, so the speaker-history comparator is date-chronological but not fully source-time verified.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite evaluation report: {output}") from None
    return report
