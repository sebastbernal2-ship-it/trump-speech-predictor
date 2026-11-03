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

## Initial results and limits

The v1 split is 16 train / 2 validation / 2 test events. On the two-event test, context Naive Bayes and the training-prevalence baseline both had micro-F1 **0.727**, macro-F1 **0.433**, mean precision@4 **0.875**, and mean recall@4 **0.75**. The context model's uncalibrated Brier score was **0.299**, worse than the baseline's **0.164**. It did not improve top-four selection on this tiny holdout; its posterior values saturated near 0 and 1 and its evidence included generic, irrelevant terms. These are weak-label diagnostics, not semantic-accuracy claims. Two events cannot establish generalization or calibration.

The frozen Qatar and Henderson context maps include assistant-authored theme drafts and turn-level outcomes. Their manually curated candidate lists are not independent benchmarks and contain no negative candidates; their 4/4 broad-topic overlap is not model performance or predictive lift.

In an exploratory replay of Henderson's frozen context, the v1 ranker (trained on the earlier 16 pilot events) ranked economy #1, election #2, foreign policy #3, immigration #4, and media #9. It missed the desired garbage-controversy/media idea from its top four and did not name specific Nevada-turnout or working-class themes. This is a concrete failure of the requested specific idea selection despite the broad-family overlap. The detailed replay and hashes are in the ignored local artifact `data/rollcall-context-theme-case-replay-v1.json`.

## Next gates

1. Improve and relevance-audit retrieval without consulting transcripts; verify event metadata and location were available before each event.
2. Add chronological pairs with both observed and absent candidate ideas.
3. Keep context-derived issue ideas distinct from broad families and evaluate those levels separately.
4. Compare against metadata-only and speaker-history baselines; calibrate only after a substantially larger independent holdout.
5. Review errors before considering more complex models or expanding beyond themes.
