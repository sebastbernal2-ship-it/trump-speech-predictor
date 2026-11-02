"""Bounded downloader for speech-like items in GovInfo's CPD collection."""

from __future__ import annotations

import argparse
from datetime import date
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from .audit import classify_title

API_ROOT = "https://api.govinfo.gov"
COLLECTION = "CPD"
SPEECH_TERMS = (
    "remarks", "address", "speech", "interview", "press conference", "conversation",
    "roundtable", "news conference", "question-and-answer", "question and answer", "q&a", "town hall",
    "ceremony", "briefing", "statement", "meeting with", "discussion",
)
TRUMP_PERIODS = ((date(2017, 1, 20), date(2021, 1, 20)), (date(2025, 1, 20), date.max))


class TextExtractor(HTMLParser):
    """Extract visible page text while ignoring markup and non-content elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "head"}:
            self.skip_depth += 1
        elif not self.skip_depth and tag.lower() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "head"} and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def read_api_key() -> str:
    key = os.environ.get("GOVINFO_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("GOVINFO_API_KEY="):
                value = stripped.split("=", 1)[1].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise ValueError("Set GOVINFO_API_KEY in the environment or in the repo-root .env file")


def _request_json(path_or_url: str, key: str) -> dict:
    parsed = urlparse(path_or_url)
    if parsed.scheme and (parsed.scheme != "https" or parsed.hostname != "api.govinfo.gov"):
        raise ValueError("GovInfo returned an unexpected pagination URL")
    path = parsed.path
    query = parse_qs(parsed.query)
    query["api_key"] = [key]
    url = API_ROOT + path + "?" + urlencode({name: values[-1] for name, values in query.items()})
    request = Request(url, headers={"User-Agent": "SpeechPrefix/0.1 research prototype", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        status = getattr(error, "code", "network error")
        raise RuntimeError(f"GovInfo API request failed ({status}); key and request URL were not displayed") from None


def iter_packages(key: str, page_size: int = 1000):
    """Page the official CPD collection endpoint. Filter all dates locally."""
    cursor = "*"
    while cursor:
        query = urlencode({"api_key": key, "pageSize": page_size, "offsetMark": cursor})
        path = f"/collections/{COLLECTION}/2017-01-20T00:00:00Z?{query}"
        result = _request_json(path, key)
        yield from result.get("packages", [])
        next_page = result.get("nextPage")
        cursor = parse_qs(urlparse(next_page).query).get("offsetMark", [""])[0] if next_page else ""
        if cursor == "*":
            raise RuntimeError("GovInfo pagination did not advance")


def _in_trump_period(issued: str, period: str = "all") -> bool:
    try:
        day = date.fromisoformat(issued[:10])
    except (TypeError, ValueError):
        return False
    periods = {"2017": TRUMP_PERIODS[:1], "2025": TRUMP_PERIODS[1:], "all": TRUMP_PERIODS}
    return any(start <= day < end for start, end in periods[period])


def _looks_like_speech(title: str) -> bool:
    lowered = title.casefold()
    return any(term in lowered for term in SPEECH_TERMS)


def clean_transcript(text: str) -> str:
    """Remove repeated publication headers that are not spoken transcript text."""
    lines = []
    for line in text.splitlines():
        if re.search(r"Administration of Donald J\. Trump", line, re.IGNORECASE):
            continue
        if re.match(r"\s*DCPD(?:\s+Number)?\b", line, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def fetch_document(package_id: str, key: str) -> str:
    # Unlike metadata endpoints, /htm returns HTML, so fetch it as bytes rather than JSON.
    query = urlencode({"api_key": key})
    request = Request(f"{API_ROOT}/packages/{package_id}/htm?{query}", headers={"User-Agent": "SpeechPrefix/0.1 research prototype"})
    try:
        with urlopen(request, timeout=45) as response:
            html = response.read().decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError):
        raise RuntimeError(f"Could not download GovInfo package {package_id}") from None
    parser = TextExtractor()
    parser.feed(html)
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    text = clean_transcript(text)
    if len(text) < 200:
        raise RuntimeError(f"GovInfo package {package_id} did not contain usable transcript text")
    return text


def collect(output: Path, limit: int, delay: float, period: str = "all") -> tuple[int, int]:
    if limit < 1 or delay < 0:
        raise ValueError("limit must be positive and delay cannot be negative")
    key = read_api_key()
    output.parent.mkdir(parents=True, exist_ok=True)
    known: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line)["speech_id"])
    candidates = []
    seen: set[str] = set()
    for item in iter_packages(key):
        package_id = item.get("packageId", "")
        issued = item.get("dateIssued", "")
        title = item.get("title", "")
        if package_id.startswith("DCPD-") and package_id not in seen and package_id not in known and _in_trump_period(issued, period) and _looks_like_speech(title):
            candidates.append(item)
            seen.add(package_id)
            if len(candidates) >= limit:
                break
    saved = failed = 0
    with output.open("a", encoding="utf-8") as stream:
        for item in candidates:
            package_id = item["packageId"]
            try:
                text = fetch_document(package_id, key)
                record = {
                    "speech_id": package_id,
                    "date": item["dateIssued"][:10],
                    "event": item.get("title", ""),
                    "source": f"https://www.govinfo.gov/content/pkg/{package_id}/",
                    "text": text,
                    "provisional_type": classify_title(item.get("title", "")),
                    "review_status": "title_only_unreviewed",
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                saved += 1
                print(f"Saved {package_id}: {record['date']} — {record['event']}")
            except (RuntimeError, ValueError, OSError) as error:
                failed += 1
                print(f"Skipped {package_id}: {error}", file=sys.stderr)
            time.sleep(delay)
    return saved, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import speech-like Trump-era records from GovInfo CPD")
    parser.add_argument("--output", default="data/speeches.jsonl")
    parser.add_argument("--limit", type=int, default=25, help="maximum new documents to download per run (default: 25)")
    parser.add_argument("--period", choices=("all", "2017", "2025"), default="all", help="Trump presidential period to collect")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between document downloads")
    args = parser.parse_args(argv)
    try:
        saved, failed = collect(Path(args.output), args.limit, args.delay, args.period)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Done: saved {saved}, skipped {failed}. Data is local and should not be committed without checking repository policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
