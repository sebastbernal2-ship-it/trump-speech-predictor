"""Provisional title-based review labels for imported presidential records."""

from __future__ import annotations

from collections import Counter


def classify_title(title: str) -> str:
    """Assign a coarse record type from the official title; this is not transcript verification."""
    value = title.casefold()
    if value.startswith("statement") or value.startswith("joint statement"):
        return "written_statement"
    if "address" in value or "united nations general assembly" in value:
        return "formal_address"
    has_remarks = "remarks" in value
    has_press = "exchange with reporters" in value or "news conference" in value
    if has_remarks and has_press:
        return "remarks_plus_press"
    if "news conference" in value or "reporters" in value:
        return "press_interaction"
    if has_remarks:
        return "event_remarks"
    return "other"


def summarize_records(records: list[dict]) -> dict[str, object]:
    counts = Counter(classify_title(record.get("event", "")) for record in records)
    return {
        "records": len(records),
        "counts": dict(sorted(counts.items())),
        "candidates_for_manual_review": sum(counts[kind] for kind in ("formal_address", "event_remarks", "remarks_plus_press")),
    }
