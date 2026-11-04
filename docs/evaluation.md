# Evaluation: context headlines → realized speech ideas

## Prediction example

One example is one speech event with:

- **Input:** news/context headline titles published in the seven days preceding the event, using only sources with dates strictly before the event date.
- **Target:** compact transcript-derived issue-phrase IDs from that event's target-speaker turns.

The target transcript is read for labels/scoring only. It must not enter its own feature construction, prompt, candidate extraction, feature vocabulary, normalization statistics, or model selection. Training on earlier examples' paired context and transcript labels is allowed and required for supervised learning. No prior speech text or history-frequency score is an input feature for this model.

## Split and loss

- Order examples by event date; keep all same-day events in one partition and one history boundary.
- Use earliest dates for training, the next chronological block for validation, and later dates for development/backtest scoring. The current corpus is small and repeatedly inspected; all scores are development results.
- Fit feature vocabularies and model parameters on training examples only. Select Bernoulli smoothing, minimum feature frequency, recent-headline cap, and logit temperature by validation binary log loss; report chronological held-out binary log loss plus P@4, recall@10, nDCG@10, candidate coverage, unseen target ideas, and per-event prediction/miss lists.
- Binary loss covers labels observed before each target. Known negative labels are included; target labels unseen in training are explicitly scored at the clipped floor (1e-12), not dropped from the denominator. The label inventory is still incomplete, so this is loss against a weak phrase ontology.
- Refit using all available eligible historical pairs for the final model. Do not call in-sample fit loss a backtest score.
- Compare against simple no-context prevalence as a reference, but do not feed its score/features into the context model.

## Current result

The current frozen snapshot union has 38 unique events; 32 have at least one extracted label. The chronological split has 24 train, 4 validation, and 4 development events. Validation chose Bernoulli Naive Bayes `alpha=4.0`, `min_df=3`, a cap of the 20 most recent deduplicated headline titles, and logit temperature `8.0`; publisher suffixes are stripped. On the four-event development slice:

| System | Binary log loss | nDCG@10 | P@4 | Recall@10 |
|---|---:|---:|---:|---:|
| Initial all-headline model | 0.635 | 0.398 | 0.313 | 0.340 |
| Updated recent-20 headline model | 0.699 | 0.462 | 0.563 | 0.343 |
| No-context prevalence reference | 0.562 | 0.439 | 0.563 | 0.315 |

The recent-20 feature variant improves nDCG@10 and recall@10 slightly versus the initial model, and raises P@4 to parity with the prevalence reference. Binary loss is worse than both the initial and no-context models. Three of four events still have an unseen target phrase; many seen phrases also rank outside the top 10. This is a mixed diagnostic, not evidence of a good model: validation and development each contain only four events.

Ignored local artifacts:

- Combined inputs: `data/rollcall-context-snapshots-retro-combined-v1.jsonl`, SHA-256 `96e94152e3b9318dbe8ab34dc28fd0d3a8f8718fa6f0ed5316cb143d43f33314`.
- Earlier report (pre-loss correction): `data/rollcall-context-to-idea-evaluation-v3.json`, SHA-256 `4f8c44f012ba34c005ffb59f27a99c2ba29391deb760150f95081e9e27accb7d`.
- Corrected-loss all-headline report/model: `data/rollcall-context-to-idea-evaluation-v4.json`, SHA-256 `e5bdb26713fe8ec8ecec885da89599d41aa3c0a80f1f25b75114a4a120c9ab7c`.
- Updated recent-headline report and serialized final fit: `data/rollcall-context-to-idea-evaluation-v5.json`, SHA-256 `fccae70aa7610f1b4dd7f4f0f3dafeee93b0ca10ad410cfb5ae2efcc1408acf0`. This is the current inference artifact and must only be applied to events after its training cutoff.

## Limitations

The snapshots were retrieved after their historical events, even when source publication dates precede the speech. They cannot establish the exact context that would have been retrieved at the time. Only 32 usable examples are available. The target extractor recognizes a small phrase inventory and misses many paraphrases, event-specific ideas, and valid outcomes. Model scores are not calibrated probabilities. Use metrics as retrospective development diagnostics only, and preserve per-event misses and hashes.
