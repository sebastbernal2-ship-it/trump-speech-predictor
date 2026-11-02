"""No-key Federal Register API search for time-bounded context evidence."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://www.federalregister.gov/api/v1/documents.json"
FIELDS = ["title", "abstract", "publication_date", "document_number", "html_url", "type", "agencies"]


def search_documents(as_of: str, query: str, limit: int = 10, lookback_days: int = 30) -> list[dict]:
    """Search only documents published in the requested pre-event time window."""
    try:
        cutoff = date.fromisoformat(as_of)
    except ValueError:
        raise ValueError("as_of must be an ISO date (YYYY-MM-DD)") from None
    if not query.strip() or not 1 <= limit <= 100 or not 1 <= lookback_days <= 365:
        raise ValueError("query is required; limit must be 1..100 and lookback_days 1..365")

    params = {
        "per_page": min(limit, 20),
        "order": "newest",
        "conditions[term]": query.strip(),
        "conditions[publication_date][gte]": (cutoff - timedelta(days=lookback_days - 1)).isoformat(),
        "conditions[publication_date][lte]": cutoff.isoformat(),
        "fields[]": FIELDS,
    }
    matches = []
    for page in range(1, (limit + 19) // 20 + 1):
        params["page"] = page
        request = Request(
            f"{API_URL}?{urlencode(params, doseq=True)}",
            headers={"User-Agent": "SpeechPrefix/0.1 research prototype", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            status = getattr(error, "code", "network error")
            raise RuntimeError(f"Federal Register API request failed ({status})") from None

        for item in payload.get("results", []):
            published = item.get("publication_date", "")
            if not published or published > cutoff.isoformat():
                continue
            agencies = [agency.get("name", "") for agency in item.get("agencies", [])]
            matches.append({
                "context_id": item.get("document_number", ""),
                "context_type": "federal_register_document",
                "published_at": published,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "") or "",
                "document_type": item.get("type", ""),
                "agencies": agencies,
                "url": item.get("html_url", ""),
                "source": "Federal Register API",
            })
            if len(matches) >= limit:
                return matches
        if len(payload.get("results", [])) < params["per_page"]:
            break
    return matches


def append_context(output: str | Path, records: list[dict]) -> int:
    """Append unseen document IDs to a local JSONL context corpus."""
    path = Path(output)
    known = set()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    known.add(json.loads(line).get("context_id"))
    path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            if not record["context_id"] or record["context_id"] in known:
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            known.add(record["context_id"])
            added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Search the no-key Federal Register API for pre-event context")
    parser.add_argument("--as-of", required=True, help="latest allowed publication date (YYYY-MM-DD)")
    parser.add_argument("--query", required=True, help="question or topic known at prediction time")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="data/world_context.jsonl")
    args = parser.parse_args(argv)
    try:
        records = search_documents(args.as_of, args.query, args.limit, args.lookback_days)
        added = append_context(args.output, records)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Found {len(records)} pre-event documents; appended {added} new records to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
