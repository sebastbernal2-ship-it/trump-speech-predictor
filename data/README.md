# Local data policy and active forecasting inputs

Collected transcripts, context snapshots, and evaluation reports are local research artifacts and remain Git-ignored. Preserve source URLs, source/event dates, retrieval timestamps, reuse terms, and content hashes where available. Do not overwrite frozen snapshots or reports; write a new versioned artifact.

## Active supervised dataset

The task is **preceding-seven-day context headlines → ideas expressed in the target speech**. The transcript corpus is `rollcall-by-type/trump-speech-candidates.jsonl`; it supplies outcome labels only. A target event's transcript and transcript-derived labels must never be features for predicting that same event. Earlier context/label pairs may train predictions for later events.

The current retrospective context union combines `rollcall-context-snapshots-topic-families-v1.jsonl` (20 events) and `rollcall-assistant-history-news-snapshots-dev-v1.jsonl` (18 distinct events) into `rollcall-context-snapshots-retro-combined-v1.jsonl` (38 events). All snapshots were retrieved after their target speeches. Publication dates do not establish when stories were available or retrieved. Thirty-two events have labels under the compact phrase extractor; six have no extracted label. See `docs/idea-layer/README.md` for frozen artifact hashes and the evaluation result.

The issue-phrase extractor yields weak, incomplete lexical labels, not exhaustive semantic outcomes. Source permission and reuse terms must be checked for each corpus; a stored note is not legal verification. No social-post archive is part of this project.

Other local data files are outside the active model unless a separately documented experiment explicitly defines and validates their use.
