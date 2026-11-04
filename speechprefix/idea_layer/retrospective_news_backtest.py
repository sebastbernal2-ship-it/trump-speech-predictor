"""Historical news-to-issue phrase replay; never a prospective input path."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from .. import headline_idea_matching as _extractor
from ..headline_idea_matching import ISSUE_PHRASE_EXTRACTOR_VERSION, extract_headline_ideas
from ..context_theme_ranker import read_jsonl


def build_retrospective_news_idea_forecasts(
    snapshots_path: str | Path,
    *,
    output_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Rank issue phrases mined from date-bounded headline snapshots.

    The snapshots may have been retrieved after their events. Only headlines
    whose feed-reported publication date is strictly before the event date are
    used; retrieval-after-event provenance is preserved and prominently marked.
    This is a retrospective backtest adapter, separate from strict snapshot APIs.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite retrospective forecast report: {output_path}")
    input_path = Path(snapshots_path)
    raw = input_path.read_bytes()
    rows = read_jsonl(input_path)
    ids = [row.get("event_id") for row in rows]
    if any(not isinstance(event_id, str) or not event_id for event_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("snapshot event IDs must be unique nonempty strings")

    forecasts = []
    for row in rows:
        event_id = row["event_id"]
        try:
            cutoff = date.fromisoformat(row.get("event_date", ""))
        except (TypeError, ValueError):
            raise ValueError(f"invalid event_date for {event_id}") from None
        sources = row.get("candidate_sources")
        if not isinstance(sources, list):
            raise ValueError(f"{event_id}: candidate_sources must be a list")
        eligible = []
        excluded = 0
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(f"{event_id}: candidate source must be an object")
            title = source.get("title")
            url = source.get("source_url") or source.get("url")
            publisher = source.get("publisher")
            published_at = source.get("published_at")
            if not all(isinstance(value, str) and value.strip() for value in (title, url, publisher, published_at)):
                raise ValueError(f"{event_id}: source requires title, URL, publisher, and published_at")
            try:
                published_date = date.fromisoformat(published_at[:10])
            except ValueError:
                excluded += 1
                continue
            if published_date >= cutoff:
                excluded += 1
                continue
            eligible.append({
                "title": title,
                "publisher": publisher,
                "published_at": published_at,
                "observed_at": source.get("observed_at"),
                "source_url": url,
            })

        extracted = extract_headline_ideas(eligible, cutoff.isoformat(), max_terms=2) if eligible else []
        candidates = []
        for idea in extracted[:top_k]:
            candidates.append({
                "rank": len(candidates) + 1,
                "idea_id": idea["idea_id"],
                "idea": idea["candidate_idea"],
                "score": idea["evidence_score"],
                "score_kind": "uncalibrated_source_support_heuristic",
                "source_count": idea["source_count"],
                "publisher_count": idea["publisher_count"],
                "supporting_sources": idea["supporting_sources"],
            })
        forecasts.append({
            "event_id": event_id,
            "event_date": cutoff.isoformat(),
            "prediction_cutoff_exclusive": cutoff.isoformat(),
            "snapshot_retrieved_at": row.get("retrieved_at") or row.get("query_retrieved_at"),
            "query_policy": row.get("query_policy"),
            "query_count": len(row.get("queries", [])) if isinstance(row.get("queries"), list) else 0,
            "raw_headline_count": len(sources),
            "eligible_pre_event_headline_count": len(eligible),
            "excluded_same_day_later_or_invalid_count": excluded,
            "predictions": candidates,
            "abstained": not candidates,
        })

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "retrospective-news-issue-phrase-forecast-replay",
        "model_family": "headline-lexicon phrase extraction ranked by source support and recency",
        "model_version": f"compact-issue-phrases-{ISSUE_PHRASE_EXTRACTOR_VERSION}",
        "evaluation_mode": "retrospective only; headlines collected after the historical events",
        "input_snapshots_sha256": hashlib.sha256(raw).hexdigest(),
        "extractor_module_sha256": hashlib.sha256(Path(_extractor.__file__).read_bytes()).hexdigest(),
        "extractor_lexicon_sha256": _extractor.issue_phrase_lexicon_sha256(),
        "query_policy": "use stored date-bounded speaker/topic-family news queries; no transcript or target-idea terms",
        "publication_cutoff_policy": "only feed-reported publication dates strictly before event date; same-day and later headlines excluded",
        "top_k": top_k,
        "forecasts": forecasts,
        "limitations": [
            "Google News snapshots were retrieved after the historical events; the stored headline dates do not prove what was searchable or observed before the speech.",
            "Publication dates, links, and publishers are feed-reported; publisher timestamps and original URLs were not independently verified.",
            "The extractor only emits phrases in a compact lexicon; it cannot generate arbitrary semantic ideas or predict whether Trump will discuss a headline.",
            "Evidence scores are uncalibrated rankings, not probabilities; repeated coverage may reflect search-query selection.",
            "No Trump-post data are used in this replay.",
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
            raise FileExistsError(f"refusing to overwrite retrospective forecast report: {path}") from None
    return report
