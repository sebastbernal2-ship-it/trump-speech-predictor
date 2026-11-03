"""Time-bounded Google News RSS adapter for the standalone idea layer.

This captures headline evidence only; it does not claim that a headline makes
Trump likely to discuss a topic or independently verify publisher timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import base64
import hashlib
import json
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fetch(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "speechprefix-idea-layer/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(2_000_000)


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field} must be a timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def retrieve_google_news_evidence(
    queries: list[str],
    cutoff_at: str,
    *,
    lookback_days: int = 30,
    timeout_seconds: float = 20,
    delay_seconds: float = 0.5,
    fetcher: Callable[[str, float], bytes] | None = None,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Capture pre-cutoff headline items and retrieval hashes for explicit queries.

    A query response retrieved at/after cutoff aborts the entire run; no partial
    source list is returned. Exact event cutoff timestamps—not day-only dates—
    determine eligibility.
    """
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("queries must be a nonempty list of nonempty strings")
    if len(set(query.strip() for query in queries)) != len(queries):
        raise ValueError("queries must be unique after trimming whitespace")
    if lookback_days < 1 or timeout_seconds <= 0 or delay_seconds < 0:
        raise ValueError("lookback_days and timeout must be positive; delay must be nonnegative")
    cutoff = _parse_timestamp(cutoff_at, "cutoff_at")
    clock = clock or (lambda: datetime.now(timezone.utc))
    fetcher = fetcher or _fetch
    if sleeper is None:
        import time
        sleeper = time.sleep
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None or now.astimezone(timezone.utc) >= cutoff:
        raise ValueError("news collection must begin strictly before prediction cutoff")

    start_date = (cutoff - timedelta(days=lookback_days)).date()
    end_date = cutoff.date() + timedelta(days=1)
    grouped: dict[str, dict[str, Any]] = {}
    query_runs = []
    for index, raw_query in enumerate(queries, 1):
        query = raw_query.strip()
        bounded_query = f"{query} after:{start_date.isoformat()} before:{end_date.isoformat()}"
        params = urlencode({"q": bounded_query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        feed_url = f"https://news.google.com/rss/search?{params}"
        body = fetcher(feed_url, timeout_seconds)
        observed = clock()
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        observed = observed.astimezone(timezone.utc)
        if observed >= cutoff:
            raise ValueError("news retrieval completed at or after prediction cutoff; no evidence returned")
        response_sha256 = hashlib.sha256(body).hexdigest()
        try:
            root = ET.fromstring(body)
        except ET.ParseError as error:
            raise ValueError(f"Google News returned invalid RSS: {error}") from None
        raw_count = 0
        eligible_count = 0
        for item in root.findall("./channel/item"):
            raw_count += 1
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            raw_pubdate = (item.findtext("pubDate") or "").strip()
            source_node = item.find("source")
            publisher = (source_node.text or "").strip() if source_node is not None else ""
            if not title or not link or not raw_pubdate or not publisher:
                continue
            try:
                published = parsedate_to_datetime(raw_pubdate)
            except (TypeError, ValueError, OverflowError):
                continue
            if published.tzinfo is None or published.utcoffset() is None:
                continue
            published = published.astimezone(timezone.utc)
            if published >= cutoff or published > observed or published.date() < start_date:
                continue
            eligible_count += 1
            canonical_identity = {
                "source_type": "news",
                "source_url": link,
                "publisher": publisher,
                "published_at": published.isoformat(),
                "headline": title,
            }
            canonical_id = "news-" + hashlib.sha256(_canonical(canonical_identity).encode("utf-8")).hexdigest()[:24]
            record = grouped.get(canonical_id)
            if record is None:
                record = {
                    "canonical_source_id": canonical_id,
                    "source_url": link,
                    "publisher": publisher,
                    "published_at": published.isoformat(),
                    "observed_at": observed.isoformat(),
                    "content": title,
                    "content_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
                    "availability_status": "verified_before_cutoff",
                    "publication_timestamp_basis": "Google News RSS pubDate; feed-reported, not independently verified with publisher",
                    "source_url_basis": "Google News RSS item link; redirect may not resolve to publisher URL",
                    "retrieval_provider": "Google News RSS",
                    "query_ids": [],
                    "query_response_sha256": [],
                }
                grouped[canonical_id] = record
            query_id = f"q{index:02d}"
            if query_id not in record["query_ids"]:
                record["query_ids"].append(query_id)
            if response_sha256 not in record["query_response_sha256"]:
                record["query_response_sha256"].append(response_sha256)
        query_runs.append({
            "query_id": f"q{index:02d}",
            "query": bounded_query,
            "feed_url": feed_url,
            "retrieved_at": observed.isoformat(),
            "response_sha256": response_sha256,
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "raw_result_count": raw_count,
            "eligible_result_count": eligible_count,
        })
        if delay_seconds and index < len(queries):
            sleeper(delay_seconds)
    finished = clock()
    if finished.tzinfo is None or finished.utcoffset() is None or finished.astimezone(timezone.utc) >= cutoff:
        raise ValueError("news collection finished at or after prediction cutoff; no evidence returned")

    sources = []
    for canonical_id, row in sorted(grouped.items()):
        observed = row["observed_at"]
        observation_key = _canonical([canonical_id, observed, sorted(row["query_response_sha256"])])
        observation_id = hashlib.sha256(observation_key.encode("utf-8")).hexdigest()[:12]
        sources.append({
            "schema_version": 1,
            "record_type": "idea-layer-source",
            "source_id": f"{canonical_id}-obs-{observation_id}",
            "canonical_source_id": canonical_id,
            "source_type": "news",
            "source_url": row["source_url"],
            "publisher": row["publisher"],
            "published_at": row["published_at"],
            "observed_at": row["observed_at"],
            "availability_status": row["availability_status"],
            "content": row["content"],
            "content_sha256": row["content_sha256"],
            "reuse_terms": (
                "Google News RSS headline metadata only; publisher-specific reuse terms not reviewed; "
                "do not redistribute and verify applicable terms before training."
            ),
            "publication_timestamp_basis": row["publication_timestamp_basis"],
            "source_url_basis": row["source_url_basis"],
            "retrieval_provider": row["retrieval_provider"],
            "query_ids": sorted(row["query_ids"]),
            "query_response_sha256": sorted(row["query_response_sha256"]),
        })
    report = {
        "schema_version": 1,
        "retrieval_type": "prospective-google-news-rss-headline-capture",
        "prediction_cutoff_at": cutoff.isoformat(),
        "lookback_days": lookback_days,
        "query_runs": query_runs,
        "unique_headline_count": len(sources),
        "sources": sources,
        "limitations": [
            "Google News RSS pubDate and item links are feed-reported; publisher timestamps/URLs were not independently resolved.",
            "This captures current headline evidence only; headlines do not establish that Trump will address the topic.",
            "Publisher-specific reuse terms must be reviewed before broader storage, training, or redistribution.",
        ],
    }
    report["retrieval_report_sha256"] = hashlib.sha256(_canonical(report).encode("utf-8")).hexdigest()
    return report
