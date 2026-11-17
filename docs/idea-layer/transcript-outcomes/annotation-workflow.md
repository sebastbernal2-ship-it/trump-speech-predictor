# Layered transcript outcomes: evidence → ideas → topics

**Status:** versioned research contract. Transcript-derived annotations are outcomes/labels, never same-event prediction inputs. The existing v1/v2 mention→idea artifacts remain immutable and retain their historical semantics; this specification defines the next annotation-system version.

## Annotation layers

1. **Transcript evidence/mention**: one passage/turn-grounded unit from the target speaker. Preserve source turn ID and exact transcript quote. Context turns may clarify but are not evidence. Mentions are the provenance anchor.
2. **Idea/occurrence**: the specific event, issue, or subject spoken about (e.g. the shooting of two National Guard members in Washington, D.C.; LGBTQ+ rights; the Ukraine war). It is not the speaker's opinion or abstract argument. One speech may discuss an occurrence through multiple passages.
3. **Key point**: a specific detail or sub-idea within an idea (e.g. a named battle within the Ukraine war). Keep its relation to the parent idea explicit.
4. **Broader topic**: a grouping of related ideas, not a replacement for their identities.
5. **Opinion/stance**: what the speaker claims, asserts, supports, opposes, proposes, requests, or forecasts about an idea/key point. Attribute all claims to the speaker; annotation does not verify truth.
6. **Emotion/tone**: separately recorded affective/tone observation attached to the relevant passage/mention, with uncertainty and evidence. Do not infer it from the idea label or conflate it with stance.

All relations are explicit and IDs resolve within the event artifact. Preserve uncertainty rather than forcing a classification. A single mention may link to multiple ideas or points when warranted. Distinguish an event/subject from claims about it: for example, an alleged fraud involving Somalians in Minnesota is an idea/occurrence, while the speaker's allegation is a separately attributed opinion/claim. The approved Washington example likewise separates the shooting, reported suspect details and troop deployment key points, and the speaker's claims/support for policy responses.

## Versioning and migration

Do not silently reinterpret existing `layered-transcript-outcome` schema v1/v2 records or the older key-point CSV contract. Implement the expanded representation as a new, explicitly versioned annotation artifact and validator. Preserve existing source records, turn IDs, outputs, and checksums. Any projection into legacy forecast idea labels must be explicit and lossy, with its mapping recorded; opinions, emotion, and transcript text are never same-event features.

## First-pass workflow and scope

Produce one assistant first-pass per applicable local target transcript, then provide it for user review/correction. This is not an independent annotation, adjudication, or objective gold reference. No two-annotator prerequisite applies. The batch runner must enumerate the corpus deterministically, write immutable per-run output, validate quotes and source turn IDs, and emit a manifest accounting for every source event as annotated, skipped with reason, or blocked with reason. Report distinct events separately from source views/records and never claim full coverage while any record is unaccounted for. Keep transcript-derived outputs in outcome-only paths and enforce the existing feature/outcome separation.

Structural exact-quote checks establish provenance only; they do not establish semantic entailment or completeness. User review remains the correction mechanism. A first pass should read the entire target transcript, annotate only target-speaker evidence, and leave unresolved boundaries, attribution, or tone explicit.

## Existing contract and limits

The legacy validator is `speechprefix/idea_layer/transcript_outcomes.py`; it validates exact quotes and mention→idea links for schemas v1/v2. Its legacy idea projection is not semantically identical to the occurrence-based idea layer above. Existing pilots and frozen point-reference materials remain unchanged. See [`layered-transcript-outcomes-v2-development.md`](layered-transcript-outcomes-v2-development.md) for historical artifact limits. No all-corpus annotation completeness or semantic accuracy claim is made until the batch manifest and outputs are reviewed.

## Run and review

```bash
base=data/source-archives/speech-content/outcomes
sources=(--source data/rollcall-events.jsonl --source data/events.jsonl)
for subset in trump-speech-candidates trump-qa-candidates trump-other-events non-trump-events; do
  sources+=(--source "data/rollcall-by-type/$subset.jsonl")
done
uv run python -m speechprefix.idea_layer.annotation_batch "${sources[@]}" --output "$base/intake-new"
# Optional: sends full packets to the named provider. Start with --limit 1.
uv run python -m speechprefix.idea_layer.annotation_generate \
  --intake "$base/intake-new" --output "$base/drafts-new" \
  --provider "$PROVIDER" --model "$MODEL" --allow-provider-transfer --limit 1
uv run python -m speechprefix.idea_layer.annotation_batch "${sources[@]}" \
  --submissions "$base/drafts-new" --output "$base/review-new"
```

Use fresh output directories. Manual assistant drafts may also be submitted as `<view_id>.json` using the shape in `annotation_generate.prompt`. Retain the original outcomes; copy drafts into a new submission directory for corrections and rerun validation. The importer marks all submissions pending user review, not gold. Provider generation failures remain in `generation-report.json`; retry them from a new intake using the review run as the pending manifest, and combine successful drafts in a fresh submission directory before collecting. Whole packets are sent without truncation; provider context limits may require manual passes on long records. No predictor consumes these artifacts.

The current structured sources cover 12,906 source rows across 5,490 distinct source event IDs. Exact duplicate views share one packet, while different transcript views remain separate. The 4,967 applicable views have target speaker text; 1,500 source rows were skipped for no target text. The flattened `trump-long-speeches.jsonl` is a derived export without turn evidence, not a seventh structured source. The first review run has one assistant draft covering three duplicate source rows; 4,966 views are pending. These are source IDs, not a cross-provider adjudication of real-world event identity.
