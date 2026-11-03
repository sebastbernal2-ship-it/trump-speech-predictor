"""Retrospective Google News RSS retrieval for exploratory context audits.

These snapshots are retrieved now, not at the historical prediction cutoff;
they must never be presented as prospective evidence.
"""

from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .context_theme_ranker import read_jsonl


ISSUE_QUERY_FAMILIES = {
    "economy_trade": "economy OR jobs OR inflation OR trade OR tariffs OR taxes",
    "immigration_border": "immigration OR border OR migrant OR asylum",
    "foreign_policy": "Iran OR Ukraine OR Israel OR Gaza OR NATO OR war",
    "domestic_policy": '"health care" OR Medicare OR education OR abortion OR crime OR police',
}


def _fetch_google_news(query: str, *, timeout: float) -> tuple[str, str, list[dict[str, Any]]]:
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    url = f"https://news.google.com/rss/search?{params}"
    request = Request(url, headers={"User-Agent": "speechprefix-context-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read(2_000_000)
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"Google News returned invalid RSS: {error}") from None
    sources = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        raw_published = (item.findtext("pubDate") or "").strip()
        if not title or not link or not raw_published:
            continue
        try:
            published = parsedate_to_datetime(raw_published)
        except (TypeError, ValueError, OverflowError):
            continue
        if published.tzinfo is None or published.utcoffset() is None:
            continue
        publisher_node = item.find("source")
        publisher = (publisher_node.text or "").strip() if publisher_node is not None else ""
        sources.append({
            "title": title,
            "publisher": publisher,
            "published_at": published.isoformat(),
            "source_url": link,
            "date_precision": "timestamp_from_google_news_rss",
        })
    return url, hashlib.sha256(body).hexdigest(), sources


def collect_speaker_news_snapshots(
    input_path: str | Path,
    output_path: str | Path,
    *,
    speaker: str = "Donald Trump",
    lookback_days: int = 30,
    delay_seconds: float = 0.4,
    timeout_seconds: float = 25,
    profile: str = "speaker-name",
) -> dict[str, Any]:
    """Fetch date-bounded speaker-only or fixed-family headline candidates.

    Queries use only the supplied speaker name, a predeclared issue-family
    vocabulary, and event dates—not transcript text or extracted target ideas.
    Since retrieval is retrospective, all results are exploratory only.
    """
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError("speaker must be a nonempty string")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    if profile not in {"speaker-name", "speaker-topic-families"}:
        raise ValueError("profile must be speaker-name or speaker-topic-families")
    if delay_seconds < 0 or timeout_seconds <= 0:
        raise ValueError("delay must be nonnegative and timeout must be positive")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing retrieval artifact: {output}")
    events = read_jsonl(input_path)
    if not events:
        raise ValueError("input must contain at least one event")
    ids = [event.get("event_id") for event in events]
    if any(not isinstance(event_id, str) or not event_id for event_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("input event IDs must be unique nonempty strings")

    rows = []
    for index, event in enumerate(events):
        event_id = event["event_id"]
        try:
            cutoff = date.fromisoformat(event.get("event_date", ""))
        except ValueError:
            raise ValueError(f"invalid event_date for {event_id}") from None
        start = cutoff - timedelta(days=lookback_days)
        query_specs = [("speaker_name", f'"{speaker.strip()}"')]
        if profile == "speaker-topic-families":
            query_specs.extend(
                (family, f'"{speaker.strip()}" {terms}')
                for family, terms in ISSUE_QUERY_FAMILIES.items()
            )
        queries = []
        sources_by_url: dict[str, dict[str, Any]] = {}
        for query_index, (purpose, query_terms) in enumerate(query_specs):
            query = f"{query_terms} after:{start.isoformat()} before:{cutoff.isoformat()}"
            retrieved_at = datetime.now(timezone.utc).isoformat()
            url, response_sha256, candidates = _fetch_google_news(query, timeout=timeout_seconds)
            eligible = [
                source for source in candidates
                if start <= date.fromisoformat(source["published_at"][:10]) < cutoff
            ]
            queries.append({
                "purpose": purpose,
                "query": query,
                "query_url": url,
                "retrieved_at": retrieved_at,
                "query_response_sha256": response_sha256,
                "raw_result_count": len(candidates),
                "eligible_result_count": len(eligible),
            })
            for source in eligible:
                source_url = source["source_url"]
                stored = sources_by_url.setdefault(source_url, {**source, "query_purposes": []})
                if purpose not in stored["query_purposes"]:
                    stored["query_purposes"].append(purpose)
            if delay_seconds and query_index + 1 < len(query_specs):
                time.sleep(delay_seconds)
        eligible = sorted(
            sources_by_url.values(),
            key=lambda source: (source["published_at"], source["title"]), reverse=True,
        )
        rows.append({
            "schema_version": 1,
            "event_id": event_id,
            "event_date": cutoff.isoformat(),
            "retrieval_protocol": f"retrospective-{profile}-search-v1",
            "retrieval_profile": profile,
            "query_policy": "speaker name, date range, and (for topic-family profile) fixed broad issue families; no transcript or target-idea terms",
            "query": queries[0]["query"],
            "query_url": queries[0]["query_url"],
            "query_retrieved_at": queries[0]["retrieved_at"],
            "query_response_sha256": queries[0]["query_response_sha256"],
            "queries": queries,
            "lookback_start_inclusive": start.isoformat(),
            "prediction_cutoff_exclusive": cutoff.isoformat(),
            "candidate_sources": eligible,
            "source_count": len(eligible),
            "exploratory_only": True,
            "prospective_warning": "Retrieved from the current search index after the event; not a historical pre-event snapshot.",
        })
        if index + 1 < len(events) and delay_seconds:
            time.sleep(delay_seconds)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing retrieval artifact: {output}") from None
    return {
        "output": str(output),
        "event_count": len(rows),
        "candidate_source_count": sum(row["source_count"] for row in rows),
        "exploratory_only": True,
        "query_policy": "speaker name/date plus fixed family terms in topic-family profile; no transcript or target-idea terms",
    }
