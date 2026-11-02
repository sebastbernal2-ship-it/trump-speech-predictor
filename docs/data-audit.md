# Local data audit

The current ignored GovInfo sample contains 1,019 CPD records dated 2017-01-20 through 2026-09-30. It is title-filtered, uneven, and contains mixed presidential documents rather than a verified speech-only corpus.

Current title-based provisional counts:

- `formal_address`: 76
- `written_statement`: 130
- `event_remarks`: 432
- `remarks_plus_press`: 329
- `press_interaction`: 40
- `other`: 12

The local paragraph segmenter produced 101,650 segments. It identified 10,900 explicit `Q.` question turns, 9,471 president-labeled turns linked to a preceding explicit question, 9,129 president-labeled remarks, and 2,787 other labeled turns. The remaining 69,363 segments have no recognized leading speaker label and remain unattributed. These are heuristic labels, not human verification; question-response links are provisional.

The segmenter trusts explicit leading labels such as `Q.` and `The President.`. It does not call every sentence with a question mark a reporter question. Review `unattributed_text` and response links before using them as supervised Q&A examples.

A no-key [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) connector stores document metadata and abstracts in ignored `data/world_context.jsonl`. The initial local query retrieved 10 documents matching `artificial intelligence`, with a publication cutoff of 2026-09-30 and a 90-day lookback. This is a connector smoke test, not a context set paired to speech targets. Use the question or topic known at prediction time as the query, and keep all retrieved publication dates at or before the prediction timestamp.

Run `speechprefix audit --input data/speeches.jsonl` for title review, `speechprefix segment` for paragraph labels, and `speechprefix context-fetch --as-of DATE --query TOPIC` for time-bounded Federal Register context. The text corpus, generated segments, world-context records, and model checkpoint remain local/ignored. Review source-specific reuse terms before expanding any collection.

Do not use the current sample for representative train/test claims. The split is chronological by date, but the corpus has not been manually reviewed for completeness, duplicates, Q&A attribution, or transcript rights.
