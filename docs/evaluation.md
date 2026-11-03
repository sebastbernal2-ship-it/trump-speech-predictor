# Evaluation protocol

## Continuation

- Keep speeches intact and split chronologically into train, validation, and test sets.
- For each test speech, reveal tokens in order. Before each reveal, score the actual next token using only the preceding prefix and training data.
- Report mean log loss, top-1 and top-k accuracy, vocabulary coverage, and results by prefix length.
- Assess calibration with reliability plots and Brier score (or an explicitly defined multiclass equivalent) before describing probabilities as confidence.
- Compare to simple frequency/n-gram baselines before testing larger models.

SpeechPlan-H continuation metrics use teacher-forced target windows and do not establish sequential online quality, calibration, or broad speech coverage. The current corpus contains mixed presidential documents, not a reviewed speech-only set. The n-gram CLI evaluator is separately minimal: it trains on all but the final record and scores that record. Both are prototype checks, not complete research protocols.

## In-speech explicit plan-path prediction

- Split by whole event dates before expanding plan labels into next-point examples.
- For each talking point, predict its theme/section/point path from only turns that ended before the first supporting target turn begins.
- Report plan-token loss, teacher-forced token accuracy, and greedy exact-path match; record the number of test examples included in greedy generation.
- For `weak-extractive` runs, report these strictly as reproduction of transcript-keyword pseudo-labels, not semantic quality or wording quality. Keep them separate from reviewed-label runs, continuation, and question-anchored response metrics.
- Use `explicit-plan-evaluate` to report per-level exact/keyword overlap, valid-path rate, output diversity, latency, event-progress slices, and majority/recent-prefix baselines.
- Report semantic idea metrics only after the labels and output mapping are human-reviewed.

Synthetic fixture scores are plumbing checks only. Real-transcript weak-label scores are still heuristic metrics; they do not establish idea-prediction accuracy. See `docs/hierarchical-speech-planner.md` for the model and plan schema.

## Question-anchored response prediction

- Keep this task distinct from ordinary continuation and before-speech forecasting.
- Use source timecodes and speaker labels. Curate the question-to-response alignment; do not assume the next span answers the question.
- Split by whole events before building examples. At each prediction point, include only spans that have ended by that time; update event state only after a span becomes observable.
- Evaluate response routing separately for target-speaker response, another speaker's response, no response, or uncertain alignment, using status accuracy and macro-F1.
- Evaluate semantic idea labels with multi-label precision/recall/F1 only when human-reviewed labels exist. Do not treat StressLens, topic, entity, or moderation annotations as gold idea labels.
- Evaluate exact wording separately with word error rate, exact word-sequence match (case-insensitive, punctuation excluded), and exact text match (case-insensitive, whitespace-normalized, punctuation-sensitive) on linked target-speaker spans.
- Report idea-label and wording metrics separately. Neither is a substitute for the other.
- Same-day external evidence is excluded when the event has no timestamp finer than a date. Assess factual support and source attribution separately; the current decoder does not emit citations.

## Pre-speech forecasting

The initial pre-speech target is recurrence of transcript-derived issue phrases in a later speech. `speaker-idea-evaluate` now evaluates a history-only ranker on the 1,990 Roll Call speech candidates, using the 1,544 events with target turns and at least one extracted issue phrase (213-phrase vocabulary). It groups by date and walks forward: 1,219 train events (through 2024-10-10), 165 validation events (2024-10-11 to 2025-12-24), and 160 test events (2026-01-06 to 2026-11-02). The half-life is selected on validation nDCG@10 from {30, 90, 180, 365, 730 days, lifetime}; the selected value was 30 days. Test metrics compare recency-weighted event prevalence with lifetime prevalence and fixed 30-/90-day windows.

On the 160-event test stream, recency weighting scored precision/recall **0.517 / 0.252 at top 4**, **0.451 / 0.524 at top 10**, and **0.293 / 0.836 at top 25**; lifetime prevalence scored **0.333 / 0.187**, **0.270 / 0.348**, and **0.243 / 0.715**. Mean nDCG@10 was **0.577 vs. 0.338**. A paired date-cluster bootstrap (117 test-date clusters, 1,000 resamples) estimated an nDCG@10 improvement over lifetime prevalence of **0.239** (95% CI **[0.197, 0.280]**). Against the stronger fixed 90-day-window baseline, the nDCG@10 difference was only **0.002** (95% CI **[-0.013, 0.015]**) and recall@10 difference **0.006** (95% CI **[-0.012, 0.023]**). Thus recency beats lifetime memory on this lexical task, but the precise decay shape has not been established as better than a simple recent-window baseline.

This is a speaker-history baseline, not a context-enabled model: it uses no headlines, external context, or exact-wording objective. Labels are automatically mined with a limited issue-phrase lexicon, not human-reviewed semantic ideas; the test is a date-ordered walk-forward and uses earlier archive transcripts as history, without verified transcript-observation timestamps. Do not interpret these metrics as semantic accuracy or context lift. The immutable local report is `data/rollcall-speaker-idea-evaluation-v3.json`.

- Use ISO-dated records and forward-only history; do not use target transcript text or post-event metadata as features.
- Compare against historical phrase-frequency and event-metadata baselines.
- Treat the initial phrase extractor and its probabilities as a baseline only: phrase candidates depend on a small stop-word list, and candidate-vocabulary Brier scores are not directly comparable across vocabularies.
- Add topic labels or event context only after specifying how they are sourced and annotated.
- Document source, permission, and transcription uncertainty.

Keep all task results separate. Continuation scores do not establish pre-speech forecasting, explicit plan-path quality, or question-anchored response quality.
