# Architecture sketch

## Implemented first

`NGramModel` is a transparent local baseline: corpus text -> tokenizer -> context-count tables -> next-word distribution. The CLI supports model training, prefix scoring, interactive chunk-by-chunk updates, and a simple held-out continuation evaluation.

## Model boundary

Each candidate model accepts a task-specific request and returns ranked candidates or scores. Keep model-specific tokenization internal and record model/version metadata. Treat scores as uncalibrated unless held-out calibration supports stronger claims.

### Continuation path

Observed transcript prefix -> candidate predictors -> optional combiner -> ranked next-token/word probabilities -> next observed chunk updates prefix.

Compare an n-gram model, an autoregressive LLM, and a concept-guided predictor. LLM token probabilities may be aggregated to words carefully; concept forecasts need an explicit mapping from concepts to measurable targets. Do not average scores blindly: validate calibration and use held-out data to fit any combiner.

### Pre-speech path

Historical dated speeches -> phrase-recurrence baseline -> ranked phrases for the next speech. The initial implementation scores phrase recurrence across prior speeches and uses walk-forward evaluation. It is a deliberately weak baseline, not a semantic topic model or a full speech generator.

A useful next version can add event context and topic-level outputs, but only after defining reliable dated event inputs and labels. Features must be frozen to what was knowable before the target speech. Keep this task's data and metrics separate from continuation.

## Data model direction

The continuation demo reader treats blank-line-separated chunks as speeches for convenience. The pre-speech baseline uses JSONL records with stable speech ID, ISO date, event, source reference, and transcript text. Question-anchored response prediction uses time-coded event JSONL with source speaker labels, page metadata, and manually reviewed response alignments. Keep transcript-quality, source, and reuse-term notes. Split whole speeches/events chronologically and exclude duplicate or near-duplicate versions across splits. The current SpeechPlan-H trainer has not yet been adapted to the time-coded event format.

## SpeechPlan-H

The primary hybrid prototype is trained from scratch. A multiscale residual vector quantizer learns future-segment codes at four temporal resolutions. A planner predicts those codes from the observed prefix, and a GPT-style autoregressive decoder cross-attends to the planned codes while generating text. Joint training combines code, quantization, and next-token losses. It does not depend on downloaded model weights. See `speechplan-hybrid.md` for the architecture and current limits; `hybrid-model.md` describes the separate pretrained candidate-fusion baseline.
