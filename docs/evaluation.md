# Evaluation: context headlines → realized speech ideas

## Prediction example

One example is one speech event with:

- **Input:** news/context headline titles published in the seven days preceding the event, using only sources with dates strictly before the event date.
- **Target:** compact transcript-derived issue-phrase IDs from that event's target-speaker turns.

The target transcript is read for labels/scoring only. It must not enter its own feature construction, prompt, candidate extraction, feature vocabulary, normalization statistics, or model selection. Training on earlier examples' paired context and transcript labels is allowed and required for supervised learning. No prior speech text or history-frequency score is an input feature for this model.

## Split and loss

- Order examples by event date; keep all same-day events in one partition and one history boundary.
- Use earliest dates for training, the next chronological block for validation, and later dates for development/backtest scoring. The current corpus is small and repeatedly inspected; all scores are development results.
- Fit feature vocabularies and model parameters on training examples only. Select Bernoulli smoothing, minimum feature frequency, headline cap/selection policy, and logit temperature by validation binary log loss; report chronological held-out binary log loss plus P@4, recall@10, nDCG@10, candidate coverage, unseen target ideas, and per-event prediction/miss lists.
- Binary loss covers labels observed before each target. Known negative labels are included; target labels unseen in training are explicitly scored at the clipped floor (1e-12), not dropped from the denominator. The label inventory is still incomplete, so this is loss against a weak phrase ontology.
- Refit using all available eligible historical pairs for the final model. Do not call in-sample fit loss a backtest score.
- Compare against simple no-context prevalence as a reference, but do not feed its score/features into the context model.

## Current result

The current frozen snapshot union has 38 unique events; 32 have at least one extracted label. The chronological split has 25 train, 3 validation, and 4 development events (grouped by whole date). A 72-configuration validation search chose Bernoulli Naive Bayes `alpha=4.0`, `min_df=3`, a cap of the 20 most recent deduplicated headline titles, and logit temperature `8.0`; publisher suffixes are stripped and capture order is preserved for date ties. On the four-event development slice:

| System | Binary log loss | nDCG@10 | P@4 | Recall@10 |
|---|---:|---:|---:|---:|
| Initial all-headline model | 0.635 | 0.398 | 0.313 | 0.340 |
| Updated recent-20 headline model (stable date-tie order) | 0.708 | 0.475 | 0.563 | 0.371 |
| No-context prevalence reference | 0.562 | 0.439 | 0.563 | 0.315 |

The recent-20 feature variant improves nDCG@10 and recall@10 versus the initial model, and raises P@4 to parity with the prevalence reference. Binary loss is worse than both the initial and no-context models. Of 45 development target IDs, 3 were absent from the preceding paired training labels and 26 additional known IDs fell outside top 10. The selected configuration's three-event validation ranking was poor (nDCG@10 0.101, P@4 0.083, recall@10 0.056), so the development lift is especially unstable. This is not evidence of a good model. See the [per-event failure analysis](idea-error-analysis.md).

Ignored local artifacts:

- Combined inputs: `data/rollcall-context-snapshots-retro-combined-v1.jsonl`, SHA-256 `96e94152e3b9318dbe8ab34dc28fd0d3a8f8718fa6f0ed5316cb143d43f33314`.
- Earlier report (pre-loss correction): `data/rollcall-context-to-idea-evaluation-v3.json`, SHA-256 `4f8c44f012ba34c005ffb59f27a99c2ba29391deb760150f95081e9e27accb7d`.
- Corrected-loss all-headline report/model: `data/rollcall-context-to-idea-evaluation-v4.json`, SHA-256 `e5bdb26713fe8ec8ecec885da89599d41aa3c0a80f1f25b75114a4a120c9ab7c`.
- Previous recent-headline report (date-tie order not preserved): `data/rollcall-context-to-idea-evaluation-v5.json`, SHA-256 `fccae70aa7610f1b4dd7f4f0f3dafeee93b0ca10ad410cfb5ae2efcc1408acf0`.
- Current recent-headline report/model with stable date-tie order and selection-policy audit: `data/rollcall-context-to-idea-evaluation-v7.json`, SHA-256 `5589eb462e6db386a972393822192b36f449ed88b5668d930442821391b5fd87`. Apply only to events after its training cutoff.

## Limitations

The snapshots were retrieved after their historical events, even when source publication dates precede the speech. They cannot establish the exact context that would have been retrieved at the time. Only 32 usable examples are available. The target extractor recognizes a small phrase inventory and misses many paraphrases, event-specific ideas, and valid outcomes. Model scores are not calibrated probabilities. Use metrics as retrospective development diagnostics only, and preserve per-event misses and hashes.
