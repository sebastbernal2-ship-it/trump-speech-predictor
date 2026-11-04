"""Post-hoc lexical stage audit for the retrospective news phrase forecaster."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..headline_idea_matching import STOPWORDS, extract_headline_ideas
from ..context_theme_ranker import read_jsonl

_GENERIC_TERMS = frozenset({
    "administration", "american", "america", "country", "federal", "government",
    "national", "people", "president", "program", "public", "state", "states",
    "support", "claim", "promote", "improve", "strengthen", "expand", "increase",
    "address", "describe", "present", "seek", "make", "provide", "protect",
    "policy", "policies", "idea", "issue", "united", "trump", "donald",
})
_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_BUCKETS = (
    "no_detectable_headline_term",
    "headline_term_but_no_phrase_candidate_overlap",
    "overlapping_phrase_candidate_below_top_k",
    "top_k_lexical_overlap_but_not_semantically_matched",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_json(path: str | Path) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    expected = value.get("report_content_sha256")
    payload = dict(value)
    payload.pop("report_content_sha256", None)
    actual = _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    if expected != actual:
        raise ValueError(f"{path}: report content hash does not validate")
    return value, _sha256(data)


def _audit_canonical(value: str) -> str:
    token = value.casefold().replace("’", "'").strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is", "ous")):
        return token[:-1]
    return token


def _terms(text: str) -> set[str]:
    terms = set()
    for match in _WORD.finditer(text):
        token = _audit_canonical(match.group())
        if len(token) >= 4 and token not in STOPWORDS and token not in _GENERIC_TERMS:
            terms.add(token)
    return terms


def audit_retrospective_news_forecast(
    *,
    snapshots_path: str | Path,
    forecast_path: str | Path,
    agreement_path: str | Path,
    reference_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Attribute uncovered reviewed ideas to observable lexical pipeline stages.

    The resulting buckets are lexical diagnostics, not semantic judgments about
    source relevance. This post-hoc function is intentionally separate from the
    prediction builder and cannot feed outcomes back into a forecast.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite news error audit: {output_path}")

    forecast, forecast_file_sha = _verified_json(forecast_path)
    agreement, agreement_file_sha = _verified_json(agreement_path)
    if agreement.get("forecast_report_sha256") != forecast_file_sha:
        raise ValueError("agreement report does not refer to the supplied forecast file")
    if forecast.get("input_snapshots_sha256") != _sha256(Path(snapshots_path).read_bytes()):
        raise ValueError("forecast report does not refer to the supplied news snapshots")

    snapshot_rows = read_jsonl(snapshots_path)
    snapshots = {row["event_id"]: row for row in snapshot_rows}
    forecasts = {row["event_id"]: row for row in forecast.get("forecasts", [])}
    if len(snapshots) != len(snapshot_rows) or len(forecasts) != len(forecast.get("forecasts", [])):
        raise ValueError("snapshot and forecast event IDs must be unique")

    references: dict[str, dict[str, Any]] = {}
    ref_hashes: dict[str, str] = {}
    for path in reference_paths:
        rows = read_jsonl(path)
        ref_hashes[str(path)] = _sha256(Path(path).read_bytes())
        for row in rows:
            event_id = row.get("event_id")
            if not isinstance(event_id, str) or event_id in references:
                raise ValueError(f"duplicate or invalid reference event ID: {event_id!r}")
            if not str(row.get("review_status", "")).startswith("assistant-reviewed-transcript-only"):
                raise ValueError(f"{event_id}: expected assistant-reviewed transcript-only reference")
            references[event_id] = row

    agreement_events = {row["event_id"]: row for row in agreement.get("events", [])}
    if set(agreement_events) - set(references):
        raise ValueError("agreement contains events without assistant-reviewed references")
    if set(agreement_events) - set(forecasts):
        raise ValueError("agreement contains events without forecasts")

    diagnostics = []
    stage_counts: Counter[str] = Counter()
    all_missed = 0
    for event_id, scored in sorted(agreement_events.items(), key=lambda item: (item[1]["event_date"], item[0])):
        snapshot = snapshots.get(event_id)
        forecast_row = forecasts[event_id]
        reference = references[event_id]
        if snapshot is None or snapshot.get("event_date") != scored.get("event_date"):
            raise ValueError(f"{event_id}: missing or date-mismatched snapshot")
        cutoff = scored["event_date"]
        eligible_sources = []
        for source in snapshot.get("candidate_sources", []):
            published = str(source.get("published_at", ""))[:10]
            if published and published < cutoff:
                eligible_sources.append({
                    "title": source["title"],
                    "publisher": source["publisher"],
                    "published_at": source["published_at"],
                    "source_url": source.get("source_url") or source.get("url"),
                })
        full_inventory = extract_headline_ideas(eligible_sources, cutoff, max_terms=2) if eligible_sources else []
        full_rank = {candidate["idea_id"]: rank for rank, candidate in enumerate(full_inventory, start=1)}
        top_candidates = forecast_row.get("predictions", [])
        top_idea_ids = {candidate["idea_id"] for candidate in top_candidates[:top_k]}
        candidate_decisions = {item["candidate_id"]: item for item in scored.get("candidate_decisions", [])}
        missed_ids = set(scored.get("outcome_idea_ids_without_match_or_partial", []))
        outcomes = {
            idea["idea_id"]: idea
            for idea in reference.get("outcome_ideas", [])
            if idea.get("forecastable") is True
        }
        if not missed_ids <= set(outcomes):
            raise ValueError(f"{event_id}: agreement lists a miss absent from the reference")

        missed_rows = []
        for idea_id in sorted(missed_ids):
            idea = outcomes[idea_id]
            label_text = f"{idea_id.replace('_', ' ')} {idea.get('idea', '')}"
            target_terms = _terms(label_text)
            headline_matches = []
            headline_term_hits: set[str] = set()
            for source in eligible_sources:
                title_terms = _terms(source["title"])
                overlap = target_terms & title_terms
                if overlap:
                    headline_term_hits.update(overlap)
                    headline_matches.append({
                        "title": source["title"],
                        "publisher": source["publisher"],
                        "published_at": source["published_at"],
                        "url": source["source_url"],
                        "overlapping_terms": sorted(overlap),
                    })
            candidate_matches = []
            for candidate in full_inventory:
                candidate_terms = _terms(candidate.get("candidate_idea", ""))
                overlap = target_terms & candidate_terms
                if overlap:
                    rank = full_rank[candidate["idea_id"]]
                    candidate_matches.append({
                        "idea_id": candidate["idea_id"],
                        "idea": candidate["candidate_idea"],
                        "rank_in_full_inventory": rank,
                        "in_top_k": candidate["idea_id"] in top_idea_ids,
                        "overlapping_terms": sorted(overlap),
                        "score": candidate["evidence_score"],
                        "source_count": candidate["source_count"],
                        "publisher_count": candidate["publisher_count"],
                    })
            candidate_matches.sort(key=lambda item: (-len(item["overlapping_terms"]), item["rank_in_full_inventory"]))
            if not headline_term_hits:
                bucket = _BUCKETS[0]
            elif not candidate_matches:
                bucket = _BUCKETS[1]
            elif any(item["in_top_k"] for item in candidate_matches):
                bucket = _BUCKETS[3]
            else:
                bucket = _BUCKETS[2]
            stage_counts[bucket] += 1
            all_missed += 1
            missed_rows.append({
                "outcome_idea_id": idea_id,
                "outcome_idea": idea.get("idea"),
                "lexical_terms_used": sorted(target_terms),
                "diagnostic_bucket": bucket,
                "headline_title_match_count": len(headline_matches),
                "headline_matched_terms": sorted(headline_term_hits),
                "headline_examples": sorted(headline_matches, key=lambda item: (-len(item["overlapping_terms"]), item["published_at"]))[:5],
                "full_inventory_phrase_match_count": len(candidate_matches),
                "best_inventory_candidates": candidate_matches[:10],
                "top_k_candidate_decision_for_lexical_matches": [
                    {
                        "idea_id": item["idea_id"],
                        "idea": item["idea"],
                        "rank": item["rank_in_full_inventory"],
                        "decision": candidate_decisions.get(item["idea_id"], {}).get("decision"),
                    }
                    for item in candidate_matches if item["in_top_k"]
                ],
            })
        diagnostics.append({
            "event_id": event_id,
            "event_date": cutoff,
            "eligible_headline_count": len(eligible_sources),
            "full_lexicon_inventory_count": len(full_inventory),
            "top_k": top_k,
            "missed_outcome_count": len(missed_rows),
            "misses_by_lexical_diagnostic_bucket": dict(Counter(row["diagnostic_bucket"] for row in missed_rows)),
            "missed_outcomes": missed_rows,
        })

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "retrospective-news-forecast-lexical-stage-audit",
        "audit_status": "post-hoc lexical diagnostic; not semantic attribution",
        "forecast_path": str(forecast_path),
        "forecast_file_sha256": forecast_file_sha,
        "snapshot_path": str(snapshots_path),
        "snapshot_file_sha256": _sha256(Path(snapshots_path).read_bytes()),
        "agreement_path": str(agreement_path),
        "agreement_file_sha256": agreement_file_sha,
        "reference_file_sha256": ref_hashes,
        "top_k": top_k,
        "scored_event_count": len(diagnostics),
        "forecastable_outcome_miss_count": all_missed,
        "misses_by_lexical_diagnostic_bucket": {bucket: stage_counts.get(bucket, 0) for bucket in _BUCKETS},
        "bucket_definitions": {
            _BUCKETS[0]: "No non-generic content token from the reviewed idea ID/summary appears in any date-filtered headline title.",
            _BUCKETS[1]: "At least one reviewed-idea token appears in a headline title, but no extracted lexicon phrase overlaps a reviewed-idea token.",
            _BUCKETS[2]: "An extracted phrase overlaps a reviewed-idea token, but all such phrases rank below top_k.",
            _BUCKETS[3]: "A top-k extracted phrase shares at least one reviewed-idea token, but the frozen assistant matching did not cover the idea; this can reflect a generic or misleading lexical overlap.",
        },
        "interpretation_limits": [
            "Token overlap is not evidence that a headline semantically discusses the reviewed idea; lexical proxies can undercount paraphrases and overcount generic matches.",
            "These buckets diagnose where literal term overlap disappears, not actual causal responsibility for each semantic miss.",
            "All inputs were retrieved after the historical events; feed dates do not prove pre-event availability.",
            "The underlying outcomes and candidate matching are assistant-reviewed references, not objective semantic ground truth.",
            "This audit must not be used to tune against the already-inspected development sample; reserve later chronological events for evaluation.",
        ],
        "events": diagnostics,
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = _sha256(canonical.encode("utf-8"))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite news error audit: {path}") from None
    return report
