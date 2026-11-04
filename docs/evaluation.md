# Evaluation: context headlines → realized speech ideas

## Prediction example

One example is one speech event with:

- **Input:** news/context headline titles published in the seven days preceding the event, using only sources with dates strictly before the event date.
- **Target:** compact transcript-derived issue-phrase IDs from that event's target-speaker turns.

The target transcript is read for labels/scoring only. It must not enter its own feature construction, prompt, candidate extraction, feature vocabulary, normalization statistics, or model selection. Training on earlier examples' paired context and transcript labels is allowed and required for supervised learning. No prior speech text or history-frequency score is an input feature for this model.

## Split and loss

- Order examples by event date; keep all same-day events in one partition and one history boundary.
- Use earliest dates for training, the next chronological block for validation, and later dates for development/backtest scoring. The current corpus is small and repeatedly inspected; all scores are development results.
- Fit feature vocabularies and model parameters on training examples only. Select hyperparameters by validation binary log loss; report held-out binary log loss plus P@4, recall@10, nDCG@10, candidate coverage, and per-event prediction/miss lists.
- Refit using all available eligible historical pairs for the final model. Do not call in-sample fit loss a backtest score.
- Compare against simple no-context prevalence as a reference, but do not feed its score/features into the context model.

## Current result

The current frozen snapshot union has 38 unique events; 32 have at least one extracted label. The chronological split has 24 train, 4 validation, and 4 development events. Validation chose Bernoulli Naive Bayes `alpha=2.0`, `min_df=1` by binary log loss. On the four-event development slice:

| System | Binary log loss | nDCG@10 | P@4 | Recall@10 |
|---|---:|---:|---:|---:|
| Preceding-week headline classifier | 3.357 | 0.428 | 0.500 | 0.315 |
| No-context prevalence reference | 0.324 | 0.439 | 0.563 | 0.315 |

The learned classifier is substantially overconfident and does not beat the reference. This is a failed first baseline, not evidence that context is unhelpful: four events, sparse labels, and highly retrospective retrieval make this comparison extremely uncertain.

Ignored local artifacts:

- Combined inputs: `data/rollcall-context-snapshots-retro-combined-v1.jsonl`, SHA-256 `96e94152e3b9318dbe8ab34dc28fd0d3a8f8718fa6f0ed5316cb143d43f33314`.
- Report: `data/rollcall-context-to-idea-evaluation-v3.json`, SHA-256 `4f8c44f012ba34c005ffb59f27a99c2ba29391deb760150f95081e9e27accb7d`.

## Limitations

The snapshots were retrieved after their historical events, even when source publication dates precede the speech. They cannot establish the exact context that would have been retrieved at the time. Only 32 usable examples are available. The target extractor recognizes a small phrase inventory and misses many paraphrases, event-specific ideas, and valid outcomes. Model scores are not calibrated probabilities. Use metrics as retrospective development diagnostics only, and preserve per-event misses and hashes.
