# Context-to-theme forecast baseline

## Task

This experiment ranks broad theme families that may appear in a future speech from dated context headlines. It is separate from transcript continuation, question-response prediction, and explicit plan-path generation. The current fixed families are elections/governance, economy/jobs, immigration, national security, public safety, health, culture/education/religion, media, energy/environment, and disasters/local resources. It does not yet generate arbitrary event-specific candidate themes such as a named controversy.

The CLI is:

```bash
speechprefix context-theme-evaluate \
  --snapshots data/rollcall-context-snapshots-v1.jsonl \
  --labels data/rollcall-theme-family-drafts-v1.jsonl \
  --output data/rollcall-context-theme-evaluation-v1.json
```

The evaluator fits one Bernoulli naive-Bayes classifier per family on the earliest 80% of event dates, reserves the next 10% for validation, and evaluates the final 10% chronologically. Inputs are source-headline tokens only; event titles, transcript text, and target theme summaries are excluded. It compares with a training-prevalence baseline and records data hashes, split IDs, predictions, evidence headlines, and metrics. Reports are immutable and the command refuses to overwrite an existing artifact.

## Pilot data and policies

`data/rollcall-context-snapshots-v1.jsonl` contains 20 metadata-only retrieval snapshots and 234 dated headline candidates. Queries used a 30-day pre-event window; same-day and later sources were excluded. Headline candidates are not relevance-reviewed. Event locations were inferred from retrospective titles, and their availability before each speech is unverified. The collection is exploratory, not a strict prospective benchmark.

`data/rollcall-theme-family-drafts-v1.jsonl` contains assistant-authored broad-family mappings from the full-transcript theme drafts. The source theme drafts and family mappings are unreviewed. Omitted families are weak inferred negatives, not verified transcript absences. All source records, draft labels, context maps, and evaluation reports remain local and Git-ignored.

## Confidence policy

The Bernoulli naive-Bayes posterior is emitted as an **uncalibrated ranking score**. It must not be presented as a reliable probability or confidence estimate until calibration and reliability are demonstrated on a substantially larger, independent chronological holdout. A saturated score is not evidence of certainty by itself; judge it by accuracy and calibration on future held-out cases.
