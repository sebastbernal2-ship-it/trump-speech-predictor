# Chunked layered-outcome extraction: long-speech prototype

**Status:** one-event engineering/development smoke test. Mechanically validated output is not semantic gold, not a forecast evaluation, and not evidence that chunking improves idea accuracy.

## Design

`research/extract_layered_transcript_outcomes_chunked.py` partitions a target-speaker transcript into contiguous, non-overlapping word-bounded chunks (default 1,800 whitespace-delimited tokens). Oversized turns are cut only at token boundaries; each artificial chunk-turn ID maps back to the original speaker-turn ID, and quote checks run against the complete source turn. The extraction model emits transcript-local mentions, local idea candidates, and typed links per chunk.

A second pass incrementally consolidates the candidate ideas: it combines the already-consolidated set with one new chunk's candidates, requires every input candidate ID to occur exactly once, and remaps links/provenance. It sees only candidate summaries and up to two linked mention paraphrases per candidate, not the full speech. This bounds each merge prompt; it cannot add/delete candidate ideas. The merge is still an AI semantic judgment and may leave duplicates or merge incorrectly.

Validated chunk results are written to immutable Git-ignored checkpoint files keyed by event, chunk input hash, provider/model, and chunk configuration. A failed merge can be retried without resending already-validated chunks. Do not stage or push the checkpoints or output artifacts.

## Smoke-test result

- Event: `dt45-19-33210f4d` (2019-12-18), 280 target-speaker turns, 18,943 words. The single-pass prototype failed validation repeatedly; that failure remains documented in [`layered-transcript-outcomes-v2-development.md`](layered-transcript-outcomes-v2-development.md).
- Chunked extraction: 11 chunks, then 10 incremental consolidation steps. The first consolidation approach tried to group all 156 local candidates in one pass and failed validation; incremental consolidation completed on retry using the saved chunk checkpoints.
- Final row passes the current schema-v2 structural validator and exact-substring checks: 294 mentions, 136 consolidated ideas, 304 typed links, and 492 exact quote citations. The model marked 4 extra citations non-matching (other exact evidence remained), normalized 4 unique token-sequence citations, and used 5 documented enum fallbacks. There are 76 ideas linked to multiple mentions.
- Final local-only artifact: `data/source-archives/layered-outcomes-chunked-v1.4-long-smoke.jsonl`, SHA-256 `ddad6dde5e87b2d27171a87c1a456481a4931ed144d15077c96a91ec4518a681`.
- Manifest: `data/source-archives/layered-outcomes-chunked-v1.4-long-smoke.jsonl.manifest.json`, SHA-256 `3a77dae3384ad1678c512eed8a323bc0b449ec4c4f10cc9de642bba89e09bcf9`.
- Provider/model: `openai-codex` / `gpt-6-luna` (mutable alias); full target-speaker transcript content was sent in multiple chunk requests. The final artifact records 44 attempts across its chunk extraction (including reused checkpoints) and 10 incremental merge steps; separate failed single-pass/global-merge experiments are not included in that count.

## Limits and next gate

Mechanical success establishes only schema/link integrity and exact quote provenance. It does not validate quote entailment, mention completeness, idea atomicity, cross-chunk identity, or forecastability. The non-overlapping boundaries can omit context; candidate-only incremental merging can leave duplicates or overmerge related claims. The 136 final nodes may be over-segmented; the 20 reductions from 156 chunk candidates are not a semantic score. No human review or independent reference inventory exists.

This tests a single long transcript only. Do not use it as training data or infer that the full 191-event 2019 corpus is covered. Next, compare independent model-as-judge diagnostics against the unchanged 49-event development slice, inspect event-level boundary cases without human annotation, and test multiple long speeches. Keep the frozen seven-day baseline and its chronological evaluation unchanged until there is a separately versioned outcome set and leakage-safe evaluation.
