"""Normalizer for user-provided, permitted Trump-post archive exports.

This module does not scrape Truth Social, X, or any other platform. It only
turns an already-acquired, provenance-bearing JSONL export into source-ledger
records.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import validate_source_evidence


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"post export {field} must be a nonempty string")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"post export {field} must be a timezone-aware RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"post export {field} must be a timezone-aware RFC 3339 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"post export {field} must be timezone-aware")
    return parsed


def load_trump_post_export(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"post export must be UTF-8: {error}") from None
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid post-export JSON on line {line_number}: {error}") from None
        if not isinstance(row, dict):
            raise ValueError(f"post-export line {line_number} must be a JSON object")
        rows.append(row)
    return normalize_trump_post_export(rows, archive_sha256=hashlib.sha256(raw).hexdigest())


def normalize_trump_post_export(
    rows: list[dict[str, Any]], *, archive_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Convert archive rows to source evidence without changing post text."""
    if not isinstance(rows, list):
        raise ValueError("post export must be a list of JSON objects")
    if archive_sha256 is not None and (
        not isinstance(archive_sha256, str) or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 hex digest")
    output = []
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"post export row {index} must be an object")
        if row.get("schema_version") != 1 or row.get("record_type") != "idea-layer-trump-post-export":
            raise ValueError(f"post export row {index} must use the v1 Trump-post export record type")
        allowed_fields = {
            "schema_version", "record_type", "platform", "account", "post_id", "permalink",
            "text", "published_at", "observed_at", "availability_status",
            "availability_evidence", "timestamp_basis", "reuse_terms",
        }
        unknown = set(row) - allowed_fields
        if unknown:
            raise ValueError(f"post export row {index} has unsupported fields: {sorted(unknown)}")
        platform = _required_text(row, "platform")
        account = _required_text(row, "account")
        post_id = _required_text(row, "post_id")
        permalink = _required_text(row, "permalink")
        parsed_permalink = urlparse(permalink)
        if parsed_permalink.scheme != "https" or not parsed_permalink.netloc:
            raise ValueError(f"post export row {index} permalink must be an HTTPS URL")
        text_value = row.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError(f"post export row {index} text must be a nonempty string")
        text = text_value
        reuse_terms = _required_text(row, "reuse_terms")
        timestamp_basis = _required_text(row, "timestamp_basis")
        availability_evidence = _required_text(row, "availability_evidence")
        published = _timestamp(row.get("published_at"), "published_at")
        observed = _timestamp(row.get("observed_at"), "observed_at")
        if published > observed:
            raise ValueError(f"post export row {index} observed_at precedes published_at")
        canonical = f"trump-post:{platform}:{account}:{post_id}"
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        observation_digest = hashlib.sha256(
            f"{canonical}\n{text_sha256}\n{observed.isoformat()}\n{archive_sha256 or ''}".encode("utf-8")
        ).hexdigest()[:12]
        source = {
            "schema_version": 1,
            "record_type": "idea-layer-source",
            "source_id": f"trump-post-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}-obs-{observation_digest}",
            "canonical_source_id": canonical,
            **({"archive_sha256": archive_sha256} if archive_sha256 else {}),
            "source_type": "trump_post",
            "source_url": permalink,
            "publisher": f"{platform}:{account}",
            "account": account,
            "post_id": post_id,
            "published_at": published.isoformat(),
            "observed_at": observed.isoformat(),
            "availability_status": row.get("availability_status"),
            "availability_evidence": availability_evidence,
            "timestamp_basis": timestamp_basis,
            "content": text,
            "content_sha256": text_sha256,
            "reuse_terms": reuse_terms,
        }
        validate_source_evidence(source)
        if source["source_id"] in seen_ids:
            raise ValueError(f"duplicate post export observation in batch: {source['source_id']}")
        seen_ids.add(source["source_id"])
        output.append(source)
    return output
