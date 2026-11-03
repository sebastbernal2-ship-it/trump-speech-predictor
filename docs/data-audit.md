# Local data audit

The current ignored GovInfo sample contains 1,019 CPD records dated 2017-01-20 through 2026-09-30. It is title-filtered, uneven, and contains mixed presidential documents rather than a verified speech-only corpus.

Current title-based provisional counts:

- `formal_address`: 76
- `written_statement`: 130
- `event_remarks`: 432
- `remarks_plus_press`: 329
- `press_interaction`: 40
- `other`: 12

The following counts describe the legacy paragraph segmenter only. These records have no per-turn timestamps and are not the canonical representation for live multi-speaker response prediction. The time-coded event schema is documented in `data/README.md`; it is now populated from Roll Call Factba.se, but its response links and idea labels have not been manually reviewed.

The paragraph segmenter produced 101,650 segments:

- 6,454 explicit `Q.` turns classified as questions.
- 4,446 explicit `Q.` turns classified as reporter prompts, not strict questions.
- 5,804 president-labeled answer paragraphs linked to a preceding strict question.
- 3,646 president-labeled responses linked to reporter prompts.
- 9,150 president-labeled remarks and 2,787 other labeled turns.
- 69,363 segments remain unattributed.

Question classification requires an explicit `Q.` label and either a question mark or a recognized interrogative opening. A named-speaker turn before the president cancels the link; the response link is still heuristic. A spot-check found clipped questions, clarifications, and floor-management turns among linked examples; do not treat the full set as verified semantic Q&A.

The question-conditioned trainer uses full 32-token answer windows. With the current v3 segmenter and chronological date split, it produces 2,790 training, 533 validation, and 437 test windows. Shorter answers are omitted from this first pass. The evaluation reports question+transcript, transcript-only, and question-only metrics separately from ordinary continuation.

## Roll Call timestamped event collection

A local collection completed on 2026-11-02 contains 5,483 events from Roll Call Factba.se's Trump archive (source index: 6,365 records). The ignored `data/rollcall-events.jsonl` has 526,503 timestamped turns and about 22.3 million whitespace-tokenized words, dated 1980-08-21 through 2026-11-02. Turn attribution is 328,210 target-speaker turns, 41,568 questioner turns, and 156,725 other-speaker turns. Of the events, 1,554 contain both Trump and a questioner; 520 contain no Trump-labeled turn, so filter by speaker role for target-specific tasks.

The importer retained video pages with nonempty text and ordered, nonnegative-duration timestamps. It excluded 450 text-only records, 369 records without timestamped transcript spans, 59 events with source timestamp anomalies, and 4 unavailable pages. The retained set has unique event IDs and no detected empty text spans or timestamp-order violations. These are automated structural checks, not manual transcript verification. `split-rollcall` derives 1,554 Q&A-candidate events, 1,990 speech-candidate events, 1,419 other Trump events, and 520 non-Trump events; category rules are defined in `data/README.md`. `prepare-rollcall-speeches` selected 1,200 long, Trump-dominant events for a continuation-only training run; preliminary metrics are in `docs/speechplan-hybrid.md`. No reviewed hierarchical speech plans exist yet; the proposed plan tree and validator are documented separately. Q&A links and idea labels remain unreviewed, and timestamped response training is not connected.

Collection permission was user-reported, but the grant details are not recorded here. Keep the data local and ignored; do not redistribute or commit it without confirming the applicable terms.

A no-key [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) connector stores document metadata and abstracts in ignored `data/world_context.jsonl`. The initial query retrieved 10 documents matching `artificial intelligence`, with a nominal cutoff of 2026-09-30 and a 90-day lookback. That first connector snapshot included same-day records; it is unpaired and must not be used for predictions dated 2026-09-30. The fetcher now excludes same-day records. A trial query about Chinese AI development returned documents with poor topical relevance, so those records were not paired or used for training. Training and inference reject evidence not published strictly before the prediction date. The current checkpoint was trained without evidence pairs, so it refuses external evidence.

Run `speechprefix audit --input data/speeches.jsonl` for title review, `speechprefix segment` to regenerate paragraph labels, and `speechprefix context-fetch --as-of DATE --query TOPIC` for time-bounded Federal Register context. The corpus, generated segments, world-context records, evidence-pair files, and checkpoints remain local/ignored. Review source-specific reuse terms before expanding collection or training.

Do not use this sample for representative train/test claims. The corpus has not been manually reviewed for completeness, duplicates, Q&A attribution, or transcript rights. The paragraph-level QA examples in these counts are not substitutes for a time-coded, manually reviewed event set.
