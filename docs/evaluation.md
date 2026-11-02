# Evaluation protocol

## Continuation

- Keep speeches intact and split chronologically into train, validation, and test sets.
- For each test speech, reveal tokens in order. Before each reveal, score the actual next token using only the preceding prefix and training data.
- Report mean log loss, top-1 and top-k accuracy, vocabulary coverage, and results by prefix length.
- Assess calibration with reliability plots and Brier score (or an explicitly defined multiclass equivalent) before describing probabilities as confidence.
- Compare to simple frequency/n-gram baselines before testing larger models.

SpeechPlan-H currently creates non-overlapping 32-token target chunks with 128-token preceding contexts, split by whole calendar dates. Its reported test loss/top-1 are teacher-forced chunk metrics; they do not establish sequential online quality, calibration, or broad speech coverage. The current corpus contains mixed presidential documents, not a reviewed speech-only set. The n-gram CLI evaluator is separately minimal: it trains on all but the final record and scores that record. Both are prototype checks, not complete research protocols.

## Pre-speech forecasting

The initial target is occurrence of recurring two-to-four-word phrases in the next speech. A walk-forward evaluation ranks phrases using only prior speeches, then scores occurrence on the next speech. It reports a candidate-phrase Brier score and mean top-10 hits.

- Use ISO-dated records and forward-only history; do not use target transcript text or post-event metadata as features.
- Compare against historical phrase-frequency and event-metadata baselines.
- Treat the initial phrase extractor and its probabilities as a baseline only: phrase candidates depend on a small stop-word list, and candidate-vocabulary Brier scores are not directly comparable across vocabularies.
- Add topic labels or event context only after specifying how they are sourced and annotated.
- Document source, permission, and transcription uncertainty.

Keep the two task results separate. A continuation score is not evidence that the system can forecast a speech before it begins.
