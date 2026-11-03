# Idea-layer record contracts

The Python validators live in `speechprefix/idea_layer/contracts.py`; tests in `tests/test_idea_layer_contracts.py` are the executable examples. Records are JSON objects; append immutable snapshots to local JSONL/artifact files rather than editing old forecasts.

## 1. Source ledger item

Required fields:

- `schema_version: 1`, `record_type: "idea-layer-source"`
- `source_id`: unique immutable observation-record ID; never recycle it
- Optional `canonical_source_id`: stable content identity used to avoid counting repeated captures of one article/post as separate evidence
- `source_type`: `news`, `trump_post`, `event_metadata`, or `other_primary_source`
- `source_url`, `publisher`, `content`, `content_sha256`, `reuse_terms`
- `published_at`, `observed_at`: timezone-aware timestamps; observed cannot precede published
- `availability_status`: `verified_before_cutoff`, `not_verified`, or `unknown`
- For `trump_post`, also keep `account` and platform `post_id`

The content hash must match the stored content. `verified_before_cutoff` is an explicit provenance assertion—not proof that the publisher’s timestamp is true. Only source items with verified status and both timestamps strictly before the forecast cutoff can be used.

## 2. Event manifest

Use `schema_version: 1`, `record_type: "idea-layer-event"`, a stable `event_id`, timezone-aware `event_start_at` and `prediction_cutoff_at`, and `event_metadata_source_ids`. The cutoff must strictly precede the scheduled start. `known_context` is a list of `{fact_id, fact, source_ids}`; every fact cites its source records. Do not put transcripts, post-event descriptions, or target labels in the manifest.

## 3. Frozen input snapshot

`idea-source-import` appends validated source records to `data/idea-layer/source-ledger.jsonl`; identical IDs are idempotent, conflicting IDs fail. `idea-news-collect` captures headline-only Google News RSS results for explicit queries and records query-response hashes, exact response bytes (base64, in the ignored local run report), and observation times. Its `pubDate` and item links are feed-reported, not publisher-verified. `idea-post-import` accepts only normalized JSONL post archives with platform/account, post ID/permalink, platform publication time, actual observation time, timestamp/availability evidence, and reuse terms; it stores the exact archive-file hash with each normalized record and does not fetch or scrape social platforms. `idea-snapshot-freeze` selects verified event metadata and eligible evidence from the requested pre-cutoff lookback, deduplicates repeated captures by `canonical_source_id`, embeds the selected source records, stores the manifest hash, and adds a content hash. Use `schema_version: 1`, `record_type: "idea-layer-input-snapshot"`, `event_id`, the exact cutoff, `snapshot_created_at`, the event manifest, unique `input_source_ids`, `sources`, and `snapshot_sha256`. A snapshot cannot be rewritten; make a new artifact for a new cutoff.

A post-export line has this shape (replace placeholders with verified values; do not mark availability verified without evidence):

```json
{"schema_version":1,"record_type":"idea-layer-trump-post-export","platform":"truth_social","account":"realDonaldTrump","post_id":"<platform-id>","permalink":"https://<platform-post-url>","text":"<exact post text>","published_at":"2030-01-20T12:30:00Z","observed_at":"2030-01-20T12:35:00Z","availability_status":"verified_before_cutoff","availability_evidence":"<archive capture or timestamp evidence>","timestamp_basis":"<platform timestamp field or archived timestamp basis>","reuse_terms":"<applicable terms or unresolved status>"}
```
## 4. Frozen pre-speech forecast

Use `schema_version: 1`, `record_type: "idea-layer-forecast"`, a unique `forecast_id`, `input_snapshot_sha256`, `event_id`, the manifest’s exact `prediction_cutoff_at`, `generated_at` (strictly before cutoff), `model_version`, `input_source_ids`, and ordered `candidates`.

Each candidate has a stable `idea_id`, readable `idea`, unique positive `rank`, numeric `score`, `score_kind`, and nonempty `evidence_source_ids` drawn from the frozen input set. `score_kind: "ranking_score"` is the default. `score_kind: "probability"` additionally requires a calibration report reference. Every event-metadata and known-context source must be in the frozen inputs. Forecast records reject transcript/outcome fields.

## 5. Append-only in-speech update

Use `schema_version: 1`, `record_type: "idea-layer-update"`, `event_id`, `forecast_id`, `updated_at`, the unique `observed_turn_ids` available at that update, and `changes`. Each change contains `idea_id`, an action (`added`, `raised`, `lowered`, `removed`, `unchanged`), and optional `evidence_turn_ids`. A cited turn must be in the visible-turn list supplied to the validator. Save each update as a new record; never rewrite the initial forecast.

The source ledger is the input to pre-speech inference. The completed transcript and post-event idea annotations belong to a separate outcome/evaluation store and must not be passed into forecast generation.
