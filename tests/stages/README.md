# Stage test ownership

Tests are organized by prediction/output granularity, matching `speechprefix/stages/`. Keep each stage's contracts, unit tests, and evaluation tests isolated; only shared pipeline tests belong at this directory's root.

- `topics/` — topic target, ranking, and topic-to-idea interface.
- `ideas/` — active direct context-to-ideas predictor and chronological evaluation.
- `sentences/` — sentence target/realization and idea-artifact interface.
- `words/` — tokenizer contract and word-sequence prediction.
- `characters/` — Unicode character-unit contract and character sequence prediction.

Currently only `ideas/` has model tests. The other folders are reserved; their absence of tests is not a pass or implementation claim.