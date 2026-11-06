"""Experimental chunked extractor for long transcript outcomes.

Each contiguous transcript chunk is extracted and validated independently. A
separate transcript-summary-only pass consolidates overlapping idea candidates;
that pass may merge candidates but cannot add or delete them. This is still an
AI-authored diagnostic, not semantic gold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from speechprefix.idea_layer.transcript_outcomes import (
    IDEA_KINDS,
    PROMINENCE_LEVELS,
    STANCES,
    validate_layered_outcome,
)

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/layered-outcomes-chunked-v1.4.jsonl"
EXTRACTOR_VERSION = "layered-transcript-outcomes-chunked-v1.4"
PROMPT_VERSION = "contiguous-chunks-then-incremental-candidate-idea-consolidation-v5"
CHUNK_CHECKPOINT_VERSION = "layered-transcript-outcomes-chunked-v1.2"
CHUNK_CHECKPOINT_PROMPT_VERSION = "contiguous-chunks-then-candidate-idea-consolidation-v3"
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_MODEL = "gpt-6-luna"

_SPEC = importlib.util.spec_from_file_location(
    "layered_outcome_single_pass",
    ROOT / "research/extract_layered_transcript_outcomes.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the single-pass extraction helpers")
_SINGLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SINGLE)


class ChunkedExtractionError(ValueError):
    """A chunk or idea consolidation could not be validated."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _partition_turns(turns: list[dict[str, str]], max_words: int) -> list[dict[str, Any]]:
    """Partition in transcript order, splitting oversized turns at whitespace tokens."""
    if max_words < 100:
        raise ValueError("max_words must be at least 100")
    segments: list[dict[str, Any]] = []
    token_pattern = re.compile(r"\S+")
    for turn_index, turn in enumerate(turns):
        turn_id, text = turn["turn_id"], turn["text"]
        spans = list(token_pattern.finditer(text))
        for part_index, start_index in enumerate(range(0, len(spans), max_words)):
            part_spans = spans[start_index:start_index + max_words]
            if not part_spans:
                continue
            start, end = part_spans[0].start(), part_spans[-1].end()
            segment_text = text[start:end]
            segments.append({
                "turn_id": f"source_{turn_index:04d}_part_{part_index:03d}",
                "source_turn_id": turn_id,
                "text": segment_text,
                "word_count": len(part_spans),
            })

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for segment in segments:
        if current and current_words + segment["word_count"] > max_words:
            chunks.append({"segments": current, "word_count": current_words})
            current, current_words = [], 0
        current.append(segment)
        current_words += segment["word_count"]
    if current:
        chunks.append({"segments": current, "word_count": current_words})
    return chunks


def _prefix_chunk_ids(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    mention_ids = {mention["mention_id"]: prefix + mention["mention_id"] for mention in row["mentions"]}
    idea_ids = {idea["idea_id"]: prefix + idea["idea_id"] for idea in row["ideas"]}
    mentions = [{**mention, "mention_id": mention_ids[mention["mention_id"]]} for mention in row["mentions"]]
    ideas = [{**idea, "idea_id": idea_ids[idea["idea_id"]]} for idea in row["ideas"]]
    links = [
        {**link, "mention_id": mention_ids[link["mention_id"]], "idea_id": idea_ids[link["idea_id"]]}
        for link in row["links"]
    ]
    return {"mentions": mentions, "ideas": ideas, "links": links}


def _consolidation_prompt(ideas: list[dict[str, Any]], mentions: list[dict[str, Any]], links: list[dict[str, str]]) -> str:
    mention_by_id = {mention["mention_id"]: mention for mention in mentions}
    candidates = []
    for idea in ideas:
        supporting = []
        for link in links:
            if link["idea_id"] == idea["idea_id"] and len(supporting) < 2:
                mention = mention_by_id[link["mention_id"]]
                supporting.append({
                    "mention_id": mention["mention_id"],
                    "relation": link["relation"],
                    "text": mention["text"],
                })
        candidates.append({
            **{key: idea[key] for key in ("idea_id", "concept", "summary", "kind", "stance", "prominence", "forecastable")},
            "member_count": len(idea.get("origin_ids", [idea["idea_id"]])),
            "linked_mentions": supporting,
        })
    return """Consolidate candidate idea nodes extracted from contiguous chunks of ONE transcript. You have only the supplied transcript-derived candidates, not the transcript or outside sources.

Group candidates only when they refer to the same independently meaningful idea expressed in different chunks. Related but distinct claims, positions, proposals, examples, or topics must remain separate. Do not merge contradictory stances. Do not invent, delete, or rewrite the substance into a new idea: every candidate ID must occur exactly once in one group. Preserve event-specific ideas even if not forecastable. When candidates differ in detail, use a concise neutral summary that all members support; use stance `unclear` if the candidates do not support a single stance. The forecastable field is an uncalibrated judgment; prefer false if candidate labels conflict. Return no more than 200 groups.

Return only this JSON shape:
{
  "groups": [
    {
      "member_idea_ids": ["candidate idea IDs"],
      "concept": "short neutral concept label",
      "summary": "one neutral sentence supported by all group members",
      "kind": "policy" | "political" | "factual_claim" | "campaign_appeal" | "ceremonial" | "personal_anecdote" | "other",
      "stance": "support" | "oppose" | "describe" | "unclear",
      "prominence": "central" | "repeated" | "secondary" | "brief",
      "forecastable": true | false
    }
  ]
}

CANDIDATE IDEAS:
""" + json.dumps(candidates, ensure_ascii=False)


def _call_with_repairs(
    prompt: str,
    *,
    provider: str,
    model: str,
    timeout: int,
    validate: Any,
    label: str,
    max_attempts: int = 5,
) -> tuple[Any, str, int]:
    errors: list[str] = []
    for attempt in range(max_attempts):
        response = ""
        try:
            response = _SINGLE._call_pi(prompt, provider=provider, model=model, timeout=timeout)
            parsed = _SINGLE._parse_object(response)
            return validate(parsed), _sha256(response.encode("utf-8")), attempt + 1
        except (ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < max_attempts:
                prompt += (
                    f"\n\nFORMAT REPAIR for {label}: {exc}. Return corrected JSON only. "
                    "Preserve valid candidate content; do not invent or omit IDs.\n"
                    "PREVIOUS RESPONSE:\n" + response
                )
    raise ChunkedExtractionError(f"{label} failed after {max_attempts} attempts: {errors}")


def _consolidate(
    value: dict[str, Any], candidate_ideas: list[dict[str, Any]], mentions: list[dict[str, Any]],
    links: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    candidate_ids = {idea["idea_id"] for idea in candidate_ideas}
    mapping: dict[str, str] = {}
    merged_ideas, provenance = [], []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{index}] must be an object")
        members = group.get("member_idea_ids")
        if not isinstance(members, list) or not members or any(not isinstance(item, str) for item in members):
            raise ValueError(f"groups[{index}].member_idea_ids must be a nonempty list of IDs")
        if len(set(members)) != len(members) or set(members) - candidate_ids:
            raise ValueError(f"groups[{index}] contains duplicate or unknown candidate IDs")
        canonical_id = sorted(members)[0]
        for member in members:
            if member in mapping:
                raise ValueError(f"candidate idea {member} occurs in multiple groups")
            mapping[member] = canonical_id
        for field in ("concept", "summary"):
            if not isinstance(group.get(field), str) or not group[field].strip():
                raise ValueError(f"groups[{index}].{field} must be nonempty text")
        if group.get("kind") not in IDEA_KINDS or group.get("stance") not in STANCES:
            raise ValueError(f"groups[{index}] has unsupported kind or stance")
        if group.get("prominence") not in PROMINENCE_LEVELS or not isinstance(group.get("forecastable"), bool):
            raise ValueError(f"groups[{index}] has unsupported prominence or forecastable value")
        member_stances = {
            idea["stance"] for idea in candidate_ideas
            if idea["idea_id"] in members and idea["stance"] != "unclear"
        }
        if len(member_stances) > 1:
            raise ValueError(f"groups[{index}] merges conflicting candidate stances")
        merged_ideas.append({
            "idea_id": canonical_id,
            "concept": group["concept"].strip(),
            "summary": group["summary"].strip(),
            "kind": group["kind"],
            "stance": group["stance"],
            "prominence": group["prominence"],
            "forecastable": group["forecastable"],
        })
        provenance.append({"idea_id": canonical_id, "member_idea_ids": sorted(members)})
    if set(mapping) != candidate_ids:
        raise ValueError(f"consolidation omitted candidate IDs: {sorted(candidate_ids - set(mapping))}")
    if len(merged_ideas) > 200:
        raise ValueError("consolidation returned more than 200 event-level ideas")
    remapped_links = []
    seen_links = set()
    for link in links:
        updated = (link["mention_id"], mapping[link["idea_id"]], link["relation"])
        if updated not in seen_links:
            seen_links.add(updated)
            remapped_links.append({"mention_id": updated[0], "idea_id": updated[1], "relation": updated[2]})
    return merged_ideas, remapped_links, provenance


def _incremental_consolidate(
    chunk_bundles: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    timeout: int,
    event_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]], int]:
    if not chunk_bundles:
        raise ChunkedExtractionError("no chunk candidates to consolidate")
    current_mentions = list(chunk_bundles[0]["mentions"])
    current_ideas = [
        {**idea, "origin_ids": [idea["idea_id"]]}
        for idea in chunk_bundles[0]["ideas"]
    ]
    current_links = list(chunk_bundles[0]["links"])
    step_records: list[dict[str, Any]] = []
    total_attempts = 0

    for bundle in chunk_bundles[1:]:
        new_ideas = [
            {**idea, "origin_ids": [idea["idea_id"]]}
            for idea in bundle["ideas"]
        ]
        candidates = current_ideas + new_ideas
        candidate_mentions = current_mentions + bundle["mentions"]
        candidate_links = current_links + bundle["links"]
        prompt = (
            "INCREMENTAL MERGE: Candidates include a previously consolidated inventory and the next transcript chunk. "
            "Preserve all existing distinct ideas and merge only true repeats.\n\n"
            + _consolidation_prompt(candidates, candidate_mentions, candidate_links)
        )
        consolidated, response_hash, attempts = _call_with_repairs(
            prompt,
            provider=provider,
            model=model,
            timeout=timeout,
            validate=lambda value: _consolidate(value, candidates, candidate_mentions, candidate_links),
            label=f"{event_id}/incremental-idea-consolidation/{bundle['chunk_id']}",
        )
        merged_ideas, merged_links, merge_provenance = consolidated
        origin_by_id = {idea["idea_id"]: idea.get("origin_ids", [idea["idea_id"]]) for idea in candidates}
        merged_by_id = {idea["idea_id"]: idea for idea in merged_ideas}
        final_id_by_temporary_id: dict[str, str] = {}
        next_ideas = []
        for group in merge_provenance:
            temporary_id = group["idea_id"]
            origins = sorted({origin for member in group["member_idea_ids"] for origin in origin_by_id[member]})
            final_id = origins[0]
            final_id_by_temporary_id[temporary_id] = final_id
            next_ideas.append({**merged_by_id[temporary_id], "idea_id": final_id, "origin_ids": origins})
        next_links = [
            {**link, "idea_id": final_id_by_temporary_id[link["idea_id"]]}
            for link in merged_links
        ]
        current_mentions = candidate_mentions
        current_ideas = next_ideas
        current_links = next_links
        total_attempts += attempts
        step_records.append({
            "added_chunk_id": bundle["chunk_id"],
            "input_candidate_count": len(candidates),
            "output_idea_count": len(current_ideas),
            "response_sha256": response_hash,
            "attempts": attempts,
        })

    final_ideas = [
        {key: value for key, value in idea.items() if key != "origin_ids"}
        for idea in current_ideas
    ]
    final_provenance = [
        {"idea_id": idea["idea_id"], "member_idea_ids": idea["origin_ids"]}
        for idea in current_ideas
    ]
    source_ids = [member for item in final_provenance for member in item["member_idea_ids"]]
    expected_ids = [idea["idea_id"] for bundle in chunk_bundles for idea in bundle["ideas"]]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(expected_ids):
        raise ChunkedExtractionError("incremental merge provenance does not cover each chunk idea exactly once")
    return final_ideas, current_links, final_provenance, step_records, total_attempts


def _extract_event(
    event: dict[str, Any],
    turns: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    timeout: int,
    max_words: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    event_id = event["event_id"]
    source_turns = {turn["turn_id"]: turn["text"] for turn in turns}
    if len(source_turns) != len(turns):
        raise ChunkedExtractionError("target-speaker turn IDs are not unique")
    chunks = _partition_turns(turns, max_words)
    if not chunks:
        raise ChunkedExtractionError("no chunkable target-speaker text")
    all_mentions: list[dict[str, Any]] = []
    all_ideas: list[dict[str, Any]] = []
    chunk_runs = []
    chunk_hashes = []
    chunk_bundles = []
    total_attempts = 0
    repair_records, rejection_records, enum_normalization_records = [], [], []
    chunk_failures = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"chunk_{chunk_index:03d}"
        segments = chunk["segments"]
        prompt_turns, blind_to_source = _SINGLE._blind_turns(
            [{"turn_id": segment["turn_id"], "text": segment["text"]} for segment in segments],
            f"chunked-case:{chunk_id}",
        )
        blind_to_source = {
            blind_id: next(segment["source_turn_id"] for segment in segments if segment["turn_id"] == piece_id)
            for blind_id, piece_id in blind_to_source.items()
        }
        prompt = (
            f"EXTRACTION SCOPE: This is contiguous chunk {chunk_index + 1} of {len(chunks)} from a larger transcript. "
            "Extract only substantive content in this chunk; do not assume it is the entire speech.\n\n"
            + _SINGLE._prompt(prompt_turns)
        )
        def validate_chunk(value: dict[str, Any]) -> dict[str, Any]:
            mapped = _SINGLE._map_layered_result(
                value,
                blind_turns=prompt_turns,
                blind_to_source=blind_to_source,
                metadata={"event_id": event_id},
                source_turns_by_id=source_turns,
            )
            if not mapped["mentions"] and not mapped["ideas"] and chunk["word_count"] > 500:
                raise ValueError("empty idea inventory for a substantive chunk")
            return mapped

        chunk_input_sha = _sha256(json.dumps(segments, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        safe_event_id = re.sub(r"[^A-Za-z0-9_.-]", "_", event_id)
        safe_provider = re.sub(r"[^A-Za-z0-9_.-]", "_", provider)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        checkpoint_path = checkpoint_dir / f"{safe_event_id}.{CHUNK_CHECKPOINT_VERSION}.{safe_provider}.{safe_model}.words{max_words}.{chunk_id}.json"
        try:
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if not isinstance(checkpoint, dict):
                    raise ChunkedExtractionError(f"checkpoint must contain an object: {checkpoint_path}")
                expected = {
                    "artifact_type": "layered-outcome-chunk-checkpoint",
                    "event_id": event_id,
                    "chunk_id": chunk_id,
                    "extractor_version": CHUNK_CHECKPOINT_VERSION,
                    "prompt_version": CHUNK_CHECKPOINT_PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "max_words": max_words,
                    "chunk_input_sha256": chunk_input_sha,
                }
                if any(checkpoint.get(key) != value for key, value in expected.items()):
                    raise ChunkedExtractionError(f"checkpoint metadata mismatch: {checkpoint_path}")
                chunk_row = checkpoint.get("row")
                row_bytes = json.dumps(chunk_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if checkpoint.get("row_sha256") != _sha256(row_bytes):
                    raise ChunkedExtractionError(f"checkpoint row hash mismatch: {checkpoint_path}")
                chunk_row = validate_layered_outcome(chunk_row, source_turns)
                response_hash, attempts = checkpoint["response_sha256"], checkpoint["attempts"]
            else:
                chunk_row, response_hash, attempts = _call_with_repairs(
                    prompt, provider=provider, model=model, timeout=timeout,
                    validate=validate_chunk, label=f"{event_id}/{chunk_id}",
                )
                row_bytes = json.dumps(chunk_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                checkpoint = {
                    "artifact_type": "layered-outcome-chunk-checkpoint",
                    "event_id": event_id,
                    "chunk_id": chunk_id,
                    "extractor_version": CHUNK_CHECKPOINT_VERSION,
                    "prompt_version": CHUNK_CHECKPOINT_PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "max_words": max_words,
                    "chunk_input_sha256": chunk_input_sha,
                    "response_sha256": response_hash,
                    "attempts": attempts,
                    "row": chunk_row,
                    "row_sha256": _sha256(row_bytes),
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("xb") as handle:
                    handle.write((json.dumps(checkpoint, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        except (ChunkedExtractionError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as exc:
            chunk_failures.append(f"{chunk_id}: {type(exc).__name__}: {exc}")
            continue
        prefix = f"{chunk_id}_"
        prefixed = _prefix_chunk_ids(chunk_row, prefix)
        all_mentions.extend(prefixed["mentions"])
        all_ideas.extend(prefixed["ideas"])
        chunk_bundles.append({"chunk_id": chunk_id, **prefixed})
        repair_records.extend(
            {**repair, "mention_id": prefix + repair["mention_id"]}
            for repair in chunk_row.get("citation_repairs", [])
        )
        rejection_records.extend(
            {**rejection, "mention_id": prefix + rejection["mention_id"]}
            for rejection in chunk_row.get("citation_rejections", [])
        )
        enum_normalization_records.extend(
            {**normalization, "object_id": prefix + normalization["object_id"]}
            for normalization in chunk_row.get("enum_normalizations", [])
        )
        chunk_hashes.append(response_hash)
        total_attempts += attempts
        chunk_runs.append({
            "chunk_id": chunk_id,
            "word_count": chunk["word_count"],
            "turn_piece_count": len(segments),
            "source_turn_ids": list(dict.fromkeys(segment["source_turn_id"] for segment in segments)),
            "mention_count": len(prefixed["mentions"]),
            "idea_count": len(prefixed["ideas"]),
            "link_count": len(prefixed["links"]),
            "response_sha256": response_hash,
            "attempts": attempts,
        })

    if chunk_failures:
        raise ChunkedExtractionError("chunk failures; validated chunks are checkpointed for resume: " + " | ".join(chunk_failures))
    if not all_ideas:
        raise ChunkedExtractionError("all chunks produced no idea candidates")
    merged_ideas, remapped_links, merge_provenance, consolidation_steps, consolidation_attempts = _incremental_consolidate(
        chunk_bundles,
        provider=provider,
        model=model,
        timeout=timeout,
        event_id=event_id,
    )
    total_attempts += consolidation_attempts
    total_words = sum(len(turn["text"].split()) for turn in turns)
    transcript = "\n".join(f"{turn['turn_id']}\t{turn['text']}" for turn in turns)
    record = {
        "schema_version": 2,
        "artifact_type": "layered-transcript-outcome",
        "event_id": event_id,
        "event_date": event.get("event_date"),
        "source_record_sha256": _SINGLE._canonical_hash(event),
        "target_transcript_sha256": _sha256(transcript.encode("utf-8")),
        "target_speaker_turn_count": len(turns),
        "target_speaker_word_count": total_words,
        "idea_definition_version": _SINGLE.IDEA_DEFINITION_VERSION,
        "extractor": {
            "version": EXTRACTOR_VERSION,
            "provider": provider,
            "model": model,
            "metadata_hidden": True,
            "news_and_weak_labels_hidden": True,
            "transcript_used_only_as_outcome": True,
            "strategy": "contiguous_chunk_extraction_then_candidate_only_idea_consolidation",
        },
        "chunking": {
            "method": "contiguous turn-order chunks with exact-source slices for oversized turns; no overlap",
            "max_words_per_chunk": max_words,
            "chunk_count": len(chunks),
            "chunks": chunk_runs,
        },
        "mentions": all_mentions,
        "ideas": merged_ideas,
        "links": remapped_links,
        "idea_merge_provenance": merge_provenance,
        "chunk_response_sha256s": chunk_hashes,
        "consolidation_steps": consolidation_steps,
        "model_attempts": total_attempts,
        "citation_repairs": repair_records,
        "citation_rejections": rejection_records,
        "enum_normalizations": enum_normalization_records,
        "coverage_note": (
            f"Extracted in {len(chunks)} contiguous non-overlapping chunks; event-level ideas were consolidated "
            "from chunk candidate summaries only. Cross-chunk identity and semantic completeness are unvalidated."
        ),
    }
    validated = validate_layered_outcome(record, source_turns)
    validated["status"] = "validated_candidate"
    validated["validation"]["normalized_citation_count"] = len(repair_records)
    validated["validation"]["enum_normalization_count"] = len(enum_normalization_records)
    validated["validation"]["rejected_citation_count"] = len(rejection_records)
    return validated


def run(
    *, events_path: Path,
    output_path: Path,
    event_ids: set[str],
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    timeout: int = 900,
    max_words: int = 1800,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    if not event_ids:
        raise ValueError("at least one --event-id is required")
    events, event_bytes = _SINGLE._read_jsonl(events_path)
    by_id = {event.get("event_id"): event for event in events}
    missing = event_ids - set(by_id)
    if missing:
        raise ValueError(f"requested event IDs are missing: {sorted(missing)}")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite chunked artifact: {output_path}")
    results = []
    checkpoint_dir = checkpoint_dir.resolve()
    for event_id in sorted(event_ids):
        event = by_id[event_id]
        turns = _SINGLE._target_turns(event)
        try:
            if not turns:
                raise ChunkedExtractionError("no target-speaker transcript turns")
            row = _extract_event(
                event, turns, provider=provider, model=model, timeout=timeout,
                max_words=max_words, checkpoint_dir=checkpoint_dir,
            )
        except (ChunkedExtractionError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as exc:
            row = {
                "schema_version": 2,
                "artifact_type": "layered-transcript-outcome-error",
                "event_id": event_id,
                "event_date": event.get("event_date"),
                "source_record_sha256": _SINGLE._canonical_hash(event),
                "status": "model_error",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(row)
        print(f"{event_id}: {row['status']} (mentions={len(row.get('mentions', []))}, ideas={len(row.get('ideas', []))})", flush=True)
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "chunked-layered-transcript-outcome-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "events_sha256": _sha256(event_bytes),
        "provider": provider,
        "model": model,
        "max_words_per_chunk": max_words,
        "checkpointing_enabled": True,
        "event_count": len(results),
        "validated_event_count": sum(row.get("status") == "validated_candidate" for row in results),
        "model_error_count": sum(row.get("status") == "model_error" for row in results),
        "mention_count": sum(len(row.get("mentions", [])) for row in results),
        "idea_count": sum(len(row.get("ideas", [])) for row in results),
        "link_count": sum(len(row.get("links", [])) for row in results),
        "quote_count": sum(row.get("validation", {}).get("evidence_quote_count", 0) for row in results),
        "enum_normalization_count": sum(row.get("validation", {}).get("enum_normalization_count", 0) for row in results),
        "outputs_local_only": True,
        "limitations": [
            "Chunking can split an idea across boundaries; the candidate-only merge is an AI judgment, not semantic gold.",
            "No overlap is used, so chunk-local extraction may miss context spanning a boundary.",
            "Exact quote checks establish provenance only; they do not establish entailment or completeness.",
            "Long target-speaker transcripts are sent to the configured provider in multiple chunk requests.",
        ],
        "output_sha256": _sha256(output_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), "events": len(results), "errors": manifest["model_error_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--event-id", action="append", dest="event_ids", required=True)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-words", type=int, default=1800)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "data/source-archives/layered-outcome-chunk-checkpoints-v1.2")
    args = parser.parse_args()
    print(json.dumps(run(
        events_path=args.events,
        output_path=args.output,
        event_ids=set(args.event_ids),
        checkpoint_dir=args.checkpoint_dir,
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
        max_words=args.max_words,
    ), indent=2))


if __name__ == "__main__":
    main()
