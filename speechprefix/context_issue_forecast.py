"""Generate source-cited issue-phrase candidates from a sealed future context."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .context_snapshots import validate_prospective_snapshot
from .headline_idea_matching import extract_headline_ideas


def generate_prospective_issue_forecast(
    snapshot_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an immutable, pre-cutoff lexical candidate forecast from v2 context."""
    snapshot_file = Path(snapshot_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing forecast: {output}")
    try:
        snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid prospective snapshot JSON: {error}") from None
    validation = validate_prospective_snapshot(snapshot)
    if not isinstance(snapshot.get("snapshot_sha256"), str):
        raise ValueError("prospective issue forecasts require a sealed snapshot hash")
    now = datetime.now(timezone.utc)
    cutoff = datetime.fromisoformat(snapshot["prediction_cutoff_at"].replace("Z", "+00:00"))
    frozen = datetime.fromisoformat(snapshot["snapshot_frozen_at"].replace("Z", "+00:00"))
    if now < frozen:
        raise ValueError("forecast generation time precedes the snapshot freeze time")
    if now >= cutoff:
        raise ValueError("cannot generate a prospective forecast at or after its prediction cutoff")

    ideas = extract_headline_ideas(
        snapshot["candidate_sources"], snapshot["prediction_cutoff_at"], max_terms=2,
    )
    ideas = [idea for idea in ideas if idea["term_count"] >= 2]
    report = {
        "schema_version": 1,
        "forecast_type": "prospective-context-issue-phrase-candidates",
        "model_family": "source-supported lexical issue phrase ranker",
        "model_version": "compact-issue-phrase-heuristic-v1",
        "task": "rank issue phrases evidenced in pre-cutoff news headlines",
        "prediction_cutoff_at": snapshot["prediction_cutoff_at"],
        "forecast_generated_at": now.isoformat(),
        "event_id": validation["event_id"],
        "retrieval_profile": snapshot.get("retrieval_profile"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "input_sha256": hashlib.sha256(snapshot_file.read_bytes()).hexdigest(),
        "candidate_policy": "compact two-term lexical issue phrases; minimum two distinct terms; no transcript or target labels",
        "score_policy": "source count, publisher diversity, and recency heuristic; not a probability or forecast confidence",
        "candidate_count": len(ideas),
        "candidates": [
            {
                "rank": rank,
                "idea": idea["candidate_idea"],
                "idea_id": idea["idea_id"],
                "evidence_score": idea["evidence_score"],
                "score_type": idea["score_type"],
                "source_count": idea["source_count"],
                "publisher_count": idea["publisher_count"],
                "supporting_sources": [
                    {
                        "title": source["title"],
                        "publisher": source["publisher"],
                        "published_at": source["published_at"],
                        "observed_at": source["observed_at"],
                        "url": source["url"],
                        "query_purposes": source.get("query_purposes", []),
                    }
                    for source in idea["supporting_sources"]
                ],
            }
            for rank, idea in enumerate(ideas, 1)
        ],
        "limitations": [
            "Lexical issue phrases are incomplete and can miss out-of-vocabulary topics or incidental matches.",
            "This ranks context-supported candidates; it does not estimate whether Trump will discuss them.",
            "The Google News RSS publication timestamp is feed-reported and not independently verified against the publisher.",
            "No context lift over the speaker-history baseline has been demonstrated.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing forecast: {output}") from None
    return report
