# Local transcript data

Do not commit collected transcripts by default. The `speechprefix import-govinfo` command can append eligible CPD records to `speeches.jsonl`. `speechprefix import-rollcall` stores permitted timestamped Factba.se events in `rollcall-events.jsonl`. You can also add permitted records manually, one JSON object per line, with these required string fields:

```json
{"speech_id":"unique-id","date":"2024-01-01","event":"event name","source":"URL or archive reference","text":"Transcript text..."}
```

Use ISO dates (`YYYY-MM-DD`) so chronological ordering is reliable. Keep source and license/permission notes. Do not include text you are not allowed to store or process. The data may be user-supplied or collected from a source with permission. Review and record source-specific reuse terms before importing or training. Roll Call's clean, timestamp-ordered video transcripts are written to `rollcall-events.jsonl`; text-only pages and pages with timestamp anomalies are excluded from this file.

## Time-coded event transcripts

Use the time-coded format for question-response forecasting. For Roll Call Factba.se data, the importer reads the public transcript-page index, fetches each event page, and preserves timestamps, speaker labels, per-span annotations, page-level metadata, source URLs, and the supplied permission note. It skips pages without timestamped transcript spans and pages with timestamp anomalies, and resumes by event ID. The lower-level parser can preserve and flag timestamp anomalies for review, but prediction-example functions reject such events. Keep this output separate from the legacy GovInfo records.

```sh
python3 -m speechprefix import-rollcall --output data/rollcall-events.jsonl
```

Use `--limit 100` for a small smoke-test batch, `--start-date`/`--end-date` to restrict event dates, and `--delay` to control request pacing. The default `--limit 0` considers all source records and retains only timestamped video pages with ordered, nonnegative-duration spans. Text-only pages, pages without transcripts, and pages with timestamp anomalies are excluded. These downloaded texts remain local/ignored.

Create disjoint, task-oriented files from the master corpus with:

```sh
python3 -m speechprefix split-rollcall
```

This writes four files under `data/rollcall-by-type/`: Q&A candidates contain Trump and questioner turns; speech candidates contain Trump turns from `Speech`, `Remarks`, `Weekly Address`, or `Convention` events with no questioner turn; other Trump events contain Trump but match neither group; non-Trump events have no Trump-labeled turn. Q&A candidates do not include reviewed question-to-answer links. The command refuses to overwrite an existing output directory. The master corpus is unchanged.

Export a target-only continuation-training set from long speech candidates with:

```sh
python3 -m speechprefix prepare-rollcall-speeches
```

By default, an event must contain at least 1,000 Trump whitespace-delimited words, with Trump contributing at least 80% of all event words. The export keeps only Trump turns and writes standard dated speech records to `trump-long-speeches.jsonl`; the full timestamped source remains in the master corpus. Use the `--min-target-words` and `--min-target-share` options to change the selection thresholds.

Human-reviewed hierarchical speech plans belong in the separate, ignored `rollcall-speech-plans.jsonl`, keyed by event ID. Each theme, section, and talking point must cite target-speaker turn IDs. Only reviewed, validated plans are eligible for the reviewed semantic-training policy; see `docs/hierarchical-speech-planner.md` for the plan tree and example format. `rollcall-plan-review-pilot.jsonl` is a local 20-speech source batch. Two blank, blinded reviewer copies are also local; follow the independent review procedure in `docs/hierarchical-speech-planner.md`. Drafts do not qualify for the reviewed semantic-training policy.

The assistant-draft plan-path and theme-only pilots, including their limits, are documented in `docs/hierarchical-speech-planner.md`; their local data and evaluation artifacts are ignored by Git.

A separate weak-label experiment can be generated from the real timestamped speech transcripts:

```sh
python3 -m speechprefix weak-plan-labels
python3 -m speechprefix explicit-plan-train \\
  --plans data/rollcall-weak-speech-plans.jsonl \\
  --label-policy weak-extractive \\
  --output checkpoints/explicit-plan-weak-run-001.pt
```

The ignored `rollcall-weak-speech-plans.jsonl` contains draft pseudo-labels: the theme uses the highest-score Factba.se/IPTC topic when available (otherwise whole-event keywords), section terms use fixed 15-turn windows, and point terms use fixed 3-turn windows. IDF weights come only from the chronological training-date partition. This is not human review, and its metrics test reproduction of source tags and keyword heuristics—not semantic idea accuracy. The trainer requires the explicit `weak-extractive` policy and records it in the model registry; it does not make these labels eligible for the reviewed-label policy.

For a manually saved page, the existing `turn-import` command accepts the source page text, including its `Full Transcript` section, and a metadata JSON file:

```sh
python3 -m speechprefix turn-import \
  --transcript /path/to/page.txt \
  --metadata /path/to/event.json \
  --output data/events.jsonl
```

Required event metadata fields are `event_id`, `event_date`, `title`, `source_url`, and `target_speaker`. Preserve `source_name` and applicable `reuse_terms` when known. Optional `target_speaker_aliases` map source name variants; `page_metadata` preserves topics, entities, moderation notes, and other page annotations. The importer preserves the raw page header, timestamped source spans, speaker names, transcript text, and per-span StressLens annotations. It does not infer question-response links.

Prediction alignments are supplied in the optional `prediction_annotations` array in the metadata JSON. Each item contains `prediction_time_seconds`, `trigger_turn_ids`, `target_turn_ids`, `response_status`, and `annotation_source`; optional fields are `outcome_turn_ids`, `response_kind`, `idea_labels`, and `example_id`. Trigger spans may come from reporters, the target speaker, or another participant—for example, a reporter's follow-up after the target asks a clarifying question. The selected spans and response links are reviewed, not inferred from adjacency. `response_status` is one of `target_spoke`, `other_speaker_spoke`, `no_target_response`, or `uncertain`. Only reviewed target-speaker spans are exact-word targets. The importer records the visible turn IDs at each prediction point and rejects examples that include target text in the observed context.

```json
{
  "event_id": "press-gaggle-2026-11-01",
  "event_date": "2026-11-01",
  "title": "Press Gaggle After Air Force One Arrival",
  "source_url": "https://example.org/transcript",
  "source_name": "Example transcript archive",
  "reuse_terms": "Review terms before use",
  "target_speaker": "Donald Trump",
  "target_speaker_aliases": ["Donald J. Trump"],
  "page_metadata": {"topics": [], "entities": [], "moderation": []},
  "prediction_annotations": [
    {
      "trigger_turn_ids": ["press-gaggle-2026-11-01:turn:0001"],
      "prediction_time_seconds": 52,
      "target_turn_ids": ["press-gaggle-2026-11-01:turn:0002"],
      "response_status": "target_spoke",
      "response_kind": "direct_answer",
      "idea_labels": ["politics.endorsement"],
      "annotation_source": "reviewed-v1"
    }
  ]
}
```

Idea labels must be human-reviewed semantic tags. They are evaluated separately from the exact transcript words. StressLens, topics, entities, and moderation labels remain source annotations; they are not treated as verified idea labels. The `prediction_example_inputs` helper separates `input`, `target`, and `provenance`; use only the `input` object as model features. Page metadata and per-span annotations are kept in provenance because they may not have been available at the prediction time.

## Standalone idea-layer evidence

`speechprefix idea-source-import --input <sources.jsonl> --ledger data/idea-layer/source-ledger.jsonl` validates and appends source records. `speechprefix idea-news-collect --manifest <event.json> --query "oil OR Gaza OR Pakistan" --report data/idea-layer/news-run.json` captures pre-cutoff Google News RSS headlines; repeat `--query` for separate searches. `speechprefix idea-snapshot-freeze --manifest <event.json> --ledger data/idea-layer/source-ledger.jsonl --output <new-snapshot.json>` freezes verified event metadata, news, and posts observed before cutoff. The ledger, reports, and snapshots are local/Git-ignored. Source rows must retain publication/observation timestamps, content hashes, source URLs as reported (or original post permalinks/IDs), and reuse terms. Google News publication timestamps and links remain feed-reported. Trump posts from a permitted, attributable archive can use `speechprefix idea-post-import --input <posts.jsonl> --ledger data/idea-layer/source-ledger.jsonl`; no platform scraper is provided. The required normalized archive row is documented in `docs/idea-layer/contracts.md`. The system does not yet rank ideas. See `docs/idea-layer/contracts.md` for exact record fields.

## Issue-phrase review sample

`rollcall-theme-review-selection-v2.jsonl` is a deterministic, quarter-stratified random sample of four events from each quarter, 2024-Q4 through 2025-Q4 (20 total), drawn from the pre-test 2024-10-11 to 2025-12-24 transcript frame. Selection seed: 240717. The two blank transcript-only reviewer packets are `rollcall-theme-review-annotator-a-v2.jsonl` and `rollcall-theme-review-annotator-b-v2.jsonl`; they omit IDs, dates, titles, URLs, predictions, and extracted issue phrases. Reviewers are prompted to list event-specific ideas before seeing the fixed family taxonomy. The taxonomy definitions are assistant-proposed and not human-adjudicated; they are not gold labels.

The coordinator-only event-ID key is `rollcall-theme-review-key-v2.json`. The separate `rollcall-theme-review-extractor-reference-v2.json` records the v2 automatic phrase labels and citations for later comparison; never share either file with reviewers before both forms are completed and locked. The sample and all review artifacts are Git-ignored. Blank forms are not annotations, and this small development sample is not a representative semantic gold set or test set. `rollcall-theme-review-assistant-drafts-v2.json` is a separate single-assistant draft: 51 event-specific ideas synthesized from up to six time-stratified, frequency-selected transcript sentences per event. Event titles were visible, the full transcripts were not read, and no human review occurred. In a post-draft mechanical check, 17/115 extractor phrases shared a cited turn with an assistant idea and 1/115 was literally contained in an idea string. These are incomplete excerpt-overlap diagnostics—not semantic precision, recall, or accuracy. The assistant draft does not fill or replace either blank reviewer packet.

The experimental selective history ranker is evaluated by `speechprefix speaker-idea-selective-evaluate`; its locked operating point is used by `speechprefix speaker-idea-forecast --as-of YYYY-MM-DD --output <new-file.json>`. The model emits at most one phrase only when it appeared in at least 50 distinct prior speech events during the preceding 90 days; otherwise it abstains. The v5 validation report is local/ignored. Its 0.839 precision is on weak lexical labels with only 18.7% event coverage, 0.017 pooled recall, and a wide [0.667, 0.952] date-cluster interval—not general or semantic accuracy. The v2 phrase rules were adjusted after 2026 test errors were inspected, so this is not a pristine target-definition evaluation. The original 2026-01-06-and-later test period is excluded before issue extraction. Historical transcript availability timestamps are not independently verified, so the rule is not proof of true real-time information availability.
