"""Collect Roll Call Factba.se transcript pages into the timestamped event schema."""

from __future__ import annotations

from datetime import date, datetime, timezone
from html.parser import HTMLParser
import gzip
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from .events import append_events, parse_timestamped_transcript


API_URL = "https://api.factsquared.com/json/factba.se-trump-20240623.php"
SOURCE_HOST = "rollcall.com"
API_HOST = "api.factsquared.com"
USER_AGENT = "SpeechPrefix/0.1 authorized transcript importer"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
TURN_ID = re.compile(r"^(?P<document>.+)-(?P<sequence>\d+)$")
TIME_RANGE = re.compile(
    r"^(?P<start>(?:\d{2}:)?\d{2}:\d{2})-"
    r"(?P<end>(?:\d{2}:)?\d{2}:\d{2})\s+\((?P<duration>[^)]*)\)$"
)
TRUMP_SPEAKER_ID = "Q22686"


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = quote(parsed.path, safe="/%:@!$&'()*+,;=-._~%")
    query = quote(parsed.query, safe="=&;%:@!$'()*+,/?-._~%")
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, parsed.fragment))


class _TranscriptHTMLParser(HTMLParser):
    """Extract source-labeled transcript spans from a Factba.se transcript page."""

    def __init__(self, document_id: str):
        super().__init__(convert_charrefs=True)
        self.document_id = document_id
        self.turn_depth = 0
        self.turn: dict | None = None
        self.capture: dict | None = None
        self.turns: list[dict] = []
        self.invalid_turns = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and self.capture and self.capture["field"] == "text":
            self.capture["parts"].append(" ")
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self.turn is None and tag == "div":
            match = TURN_ID.fullmatch(attributes.get("id") or "")
            if match and match.group("document") == self.document_id:
                self.turn = {
                    "source_turn_id": attributes["id"],
                    "source_sequence": int(match.group("sequence")),
                    "speaker_name": "",
                    "time_range": "",
                    "annotation": "",
                    "text": "",
                }
                self.turn_depth = 1
                return
        elif self.turn is not None and tag == "div":
            self.turn_depth += 1

        if self.turn is None or self.capture is not None:
            return
        if tag == "h2" and not self.turn["speaker_name"]:
            self._begin_capture("speaker_name", tag)
        elif tag == "span" and not self.turn["time_range"]:
            self._begin_capture("time_range", tag)
        elif tag == "div" and not self.turn["annotation"] and {
            "text-center", "rounded-md"
        }.issubset(classes):
            self._begin_capture("annotation", tag)
        elif tag == "div" and {"flex-auto", "leading-loose"}.issubset(classes):
            self._begin_capture("text", tag)

    def handle_endtag(self, tag: str) -> None:
        if self.turn is None:
            return
        if self.capture and tag == self.capture["tag"] and (
            tag != "div" or self.turn_depth == self.capture["depth"]
        ):
            value = " ".join("".join(self.capture["parts"]).split())
            self.turn[self.capture["field"]] = value
            self.capture = None
        if tag == "div":
            self.turn_depth -= 1
            if self.turn_depth == 0:
                self._finish_turn()

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.capture["parts"].append(data)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and self.capture and self.capture["field"] == "text":
            self.capture["parts"].append(" ")
        else:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

    def _begin_capture(self, field: str, tag: str) -> None:
        self.capture = {
            "field": field,
            "tag": tag,
            "depth": self.turn_depth,
            "parts": [],
        }

    def _finish_turn(self) -> None:
        assert self.turn is not None
        if not self.turn["speaker_name"] or not self.turn["time_range"] or not self.turn["text"]:
            self.invalid_turns += 1
        else:
            self.turns.append(self.turn)
        self.turn = None
        self.capture = None


def _request(url: str, *, method: str = "GET", timeout: int = 30) -> bytes:
    host = (urlparse(url).hostname or "").lower()
    if host not in {SOURCE_HOST, API_HOST}:
        raise ValueError(f"refusing non-source URL: {url}")
    request = Request(
        _normalize_url(url),
        data=b"" if method == "POST" else None,
        method=method,
        headers={
            "Accept": "application/json, text/html",
            "Accept-Encoding": "gzip",
            "User-Agent": USER_AGENT,
        },
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                final_host = (urlparse(response.geturl()).hostname or "").lower()
                if final_host not in {SOURCE_HOST, API_HOST}:
                    raise ValueError(f"source redirected to an unexpected host: {final_host}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("source response exceeds the 32 MiB safety limit")
                return raw
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise RuntimeError(f"HTTP {error.code} from {url} after retries") from error
            wait = error.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 1.0 * (attempt + 1))
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("source request failed after retries")


def iter_factbase_records(*, timeout: int = 30, delay: float = 0.15):
    """Yield the source's complete Trump transcript-page index in API order."""
    page = 1
    total_pages = None
    seen = set()
    while total_pages is None or page <= total_pages:
        query = urlencode({"page": page})
        payload = json.loads(_request(f"{API_URL}?{query}", method="POST", timeout=timeout))
        meta = payload.get("meta", {})
        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise ValueError(f"Factba.se API page {page} has no record list")
        total_pages = int(meta.get("total_pages", 0))
        if total_pages < 1:
            return
        for record in rows:
            document_id = record.get("document_id")
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            yield record
        page += 1
        if page <= total_pages and delay:
            time.sleep(delay)


def parse_factbase_page(page_html: str, record: dict, *, reuse_terms: str) -> dict | None:
    """Convert one source page into the shared time-coded event representation."""
    document_id = record.get("document_id")
    page_url = record.get("factbase_url")
    if "no transcript" in (record.get("record_title") or "").casefold():
        return None
    if not isinstance(document_id, str) or not isinstance(page_url, str):
        raise ValueError("Factba.se record is missing its document ID or transcript URL")
    source_url = _normalize_url(page_url)
    parsed_url = urlparse(source_url)
    if parsed_url.hostname != SOURCE_HOST or not parsed_url.path.startswith("/factbase/trump/transcript/"):
        raise ValueError(f"unexpected Factba.se transcript URL: {page_url}")

    page_parser = _TranscriptHTMLParser(document_id)
    page_parser.feed(page_html)
    page_parser.close()
    if not page_parser.turns or page_parser.invalid_turns:
        return None

    lines = ["Roll Call Factba.se", record.get("record_title", ""), "Full Transcript"]
    for turn in page_parser.turns:
        match = TIME_RANGE.fullmatch(turn["time_range"])
        if not match:
            raise ValueError(f"invalid source time range in {turn['source_turn_id']}: {turn['time_range']!r}")
        lines.append(f"{turn['speaker_name']} {turn['time_range']}")
        if turn["annotation"]:
            lines.append(turn["annotation"])
        lines.append(turn["text"])

    speakers = (record.get("document") or {}).get("speakers") or []
    speaker_ids = {
        speaker.get("speaker", "").casefold(): speaker.get("speaker_id")
        for speaker in speakers
        if speaker.get("speaker") and speaker.get("speaker_id")
    }
    aliases = {"Donald Trump", "Donald J. Trump", "President Trump", "The President"}
    aliases.update(
        speaker["speaker"] for speaker in speakers
        if speaker.get("speaker_id") == TRUMP_SPEAKER_ID and speaker.get("speaker")
    )
    source_date = record.get("date", "")
    event_date = source_date[:10] if isinstance(source_date, str) else ""
    metadata = {
        "event_id": document_id,
        "event_date": event_date,
        "title": record.get("record_title") or document_id,
        "source_url": source_url,
        "source_name": "Roll Call Factba.se",
        "reuse_terms": reuse_terms,
        "target_speaker": "Donald Trump",
        "target_speaker_aliases": sorted(aliases),
        "page_metadata": {
            "factbase_record": record,
            "collector": {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": API_URL,
            },
        },
    }
    try:
        event = parse_timestamped_transcript(
            "\n".join(lines), metadata, allow_timestamp_anomalies=True
        )
    except ValueError as error:
        raise ValueError(f"{document_id}: {error}") from error
    for parsed_turn, source_turn in zip(event["turns"], page_parser.turns):
        parsed_turn["source_turn_id"] = source_turn["source_turn_id"]
        parsed_turn["source_sequence"] = source_turn["source_sequence"]
        speaker_id = speaker_ids.get(source_turn["speaker_name"].casefold())
        if speaker_id:
            parsed_turn["speaker_id"] = speaker_id
    return event


def _existing_event_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    with output.open(encoding="utf-8") as stream:
        return {json.loads(line)["event_id"] for line in stream if line.strip()}


def collect_rollcall(
    output_path: str | Path = "data/rollcall-events.jsonl",
    *,
    limit: int = 0,
    delay: float = 0.15,
    timeout: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    reuse_terms: str = "User-reported permission for bulk retrieval and ML use; grant details are not stored here.",
    progress=None,
) -> dict[str, object]:
    """Fetch all available time-coded Factba.se pages, resuming by event ID."""
    if limit < 0 or delay < 0 or timeout <= 0:
        raise ValueError("limit and delay must be nonnegative; timeout must be positive")
    lower = date.fromisoformat(start_date) if start_date else None
    upper = date.fromisoformat(end_date) if end_date else None
    if lower and upper and lower > upper:
        raise ValueError("start_date must be on or before end_date")

    existing = _existing_event_ids(output_path)
    stats: dict[str, object] = {
        "records_seen": 0, "pages_fetched": 0, "events_added": 0,
        "no_timestamped_transcript": 0, "unavailable_source_page": 0,
        "unavailable_source_page_ids": [], "nonvideo_media": 0,
        "timestamp_anomaly_events": 0, "already_present": 0,
        "outside_date_range": 0,
    }

    def events():
        for record in iter_factbase_records(timeout=timeout, delay=delay):
            if limit and stats["records_seen"] >= limit:
                break
            stats["records_seen"] += 1
            event_id = record["document_id"]
            if event_id in existing:
                stats["already_present"] += 1
                continue
            if "no transcript" in (record.get("record_title") or "").casefold():
                stats["no_timestamped_transcript"] += 1
                continue
            if record.get("media_type") != "Video":
                stats["nonvideo_media"] += 1
                continue
            raw_date = record.get("date", "")
            try:
                record_date = date.fromisoformat(raw_date[:10])
            except (TypeError, ValueError):
                raise ValueError(f"invalid event date for {event_id}: {raw_date!r}") from None
            if (lower and record_date < lower) or (upper and record_date > upper):
                stats["outside_date_range"] += 1
                continue
            try:
                page_html = _request(record["factbase_url"], timeout=timeout).decode("utf-8", "replace")
            except HTTPError as error:
                if error.code not in {404, 410}:
                    raise
                stats["unavailable_source_page"] += 1
                stats["unavailable_source_page_ids"].append({
                    "event_id": event_id, "url": record["factbase_url"],
                    "reason": f"HTTP {error.code}",
                })
                if delay:
                    time.sleep(delay)
                continue
            except RuntimeError as error:
                message = str(error)
                if not message.startswith(("HTTP 500 from ", "HTTP 502 from ")):
                    raise
                stats["unavailable_source_page"] += 1
                stats["unavailable_source_page_ids"].append({
                    "event_id": event_id, "url": record["factbase_url"],
                    "reason": message.split(" from ", 1)[0],
                })
                if delay:
                    time.sleep(delay)
                continue
            stats["pages_fetched"] += 1
            event = parse_factbase_page(page_html, record, reuse_terms=reuse_terms)
            if event is None:
                stats["no_timestamped_transcript"] += 1
            else:
                if event["timestamp_anomalies"]:
                    stats["timestamp_anomaly_events"] += 1
                    continue
                stats["events_added"] += 1
                yield event
            if delay:
                time.sleep(delay)
            if progress and stats["records_seen"] % 100 == 0:
                progress(dict(stats))

    appended = append_events(output_path, events())
    stats["events_added"] = appended
    return stats
