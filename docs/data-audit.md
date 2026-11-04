# Data audit and current limits

## Active historical input/output pairs

- Transcript outcomes: `data/rollcall-by-type/trump-speech-candidates.jsonl`, 1,990 dated speech candidates from 1988–2026.
- Historical context: 20 `topic-families` snapshots plus 18 news snapshots for distinct events (38 paired events).
- Usable labels: 32 events have at least one issue phrase under `compact-issue-phrases-v2`; six are skipped due to no extracted phrase.
- Input window in the active model: source publication date in the seven days before the event through the previous day.
- Snapshot timing: all snapshots were collected after the historical target events. Historical article publication dates do not establish when an article was available or retrieved.

## Outcome-label limits

The target phrase extractor is a fixed lexical inventory. It can miss ideas with other wording, novel concepts, paraphrases, broad propositions, and short/ceremonial content. A phrase present in the transcript is not a complete semantic inventory. Current metrics evaluate agreement with these labels only. Assistant-reviewed idea references can diagnose omissions but are interpretive and not objective semantic gold; use the assistant-only policy when reviewing.

## Active artifacts

The frozen 38-event snapshot union and classifier report are Git-ignored local artifacts. See `docs/idea-layer/README.md` for their SHA-256 hashes, split, scores, and limitations. Neither the feature contract nor the current result should be generalized beyond this experiment.
