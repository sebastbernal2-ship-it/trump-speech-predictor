# SpeechPlan-H

SpeechPlan-H is a from-scratch, word-level model. It does not load GPT-2 or Stanford NCM weights. It uses the installed Transformers GPT-2 implementation only as a randomly initialized autoregressive decoder.

## Model inputs

The current model has separate channels for:

- **Transcript context:** observed text before or around the target response.
- **Question:** explicit question text, supplied separately from transcript context.
- **Dated evidence:** optional source-attributed text published before the prediction point. The current Federal Register sample is unpaired and is not model input.

A planner predicts coarse-to-fine residual VQ codes for the next segment. The autoregressive decoder cross-attends to the plan and question/evidence memory. A training-only future encoder and residual vector quantizer learn discrete codes at 1, 4, 16, and 32 positions over a 32-token target.

## Time-coded event data

The target interaction is a timestamped, multi-speaker event, not an adjacent paragraph pair. `speechprefix/events.py` parses source spans and keeps their timestamps, speaker names, text, page metadata, and per-span annotations. The `turn-import` CLI writes these events as JSONL; see `data/README.md` for the schema. The separate explicit theme/section/talking-point path model is described in `docs/hierarchical-speech-planner.md`; it does not alter or condition this SpeechPlan-H decoder.

Question-to-response alignments are supplied by a reviewer. They identify the prediction cutoff, visible trigger spans, any target-speaker response spans, other-speaker outcomes, and optional human-reviewed idea labels. The parser does not guess that a nearby span answers a question. `prediction_example_inputs` separates causal model inputs, gold targets, and source provenance. It constructs a transcript prefix and separate trigger text from visible spans. Page and StressLens annotations are preserved under provenance, not silently treated as information available to the model. Response status, semantic idea labels, and exact target words remain distinct.

**The existing trainer has not yet been adapted to this time-coded event schema.** Its question-response dataset is the older paragraph-based prototype, with noisy heuristic links; its checkpoint metrics are not evidence of performance on multi-speaker live interactions. The new event schema and metric functions provide a safer data/evaluation foundation, not a trained live-response model.

## Evaluation

Split by whole events, and reveal transcript spans only after their end time. Keep before-speech forecasting, ordinary continuation, and question-anchored response prediction separate. Evaluate response routing, human-labeled ideas, and target-speaker wording independently:

- Idea labels: multi-label precision, recall, F1, and exact-set match on reviewed labels only.
- Exact wording: word error rate, exact word-sequence match, and exact text match on linked target-speaker spans.
- Response routing: target speaker responded, another speaker responded, no response, or uncertain alignment.

`turn_evaluation.py` implements routing, idea, and wording metrics, but these metrics are not wired into SpeechPlan-H training yet. StressLens, topic, entity, and moderation annotations are source metadata, not verified idea labels. See `docs/evaluation.md` for the protocol.

## Legacy paragraph experiment and evidence

`hybrid-train` accepts dated speech records and uses chronological whole-date splits. It can train continuation from the target-only `trump-long-speeches.jsonl` export. Its optional Q&A windows still use the legacy paragraph segmenter; timestamped Q&A events are not connected to response training. Legacy Q&A links require manual review and may be invalidated by an intervening named speaker.

The Federal Register fetcher stores dated metadata and abstracts without an API key. Documents are not joined to questions automatically. Same-day evidence is excluded because source transcripts do not provide event timestamps. Evidence-pair training remains separate from timestamped response alignment; evidence is not accepted unless a checkpoint records training on paired evidence. The decoder does not emit citations, so factual support must be evaluated separately.

## Training and generation

```sh
.venv/bin/python -m speechprefix hybrid-train \
  --input data/speeches.jsonl --output checkpoints/speechplan-h-run-001.pt --epochs 1 \
  --registry models/registry.jsonl
.venv/bin/python -m speechprefix hybrid \
  --model checkpoints/speechplan-h.pt \
  --prefix "We have been discussing our plans" \
  --question "What is the plan?"
```

Use `--max-steps` only for plumbing checks. Training refuses to overwrite an existing checkpoint and appends its run provenance, data/source hashes, and artifact hash to `models/registry.jsonl`. See `docs/model-versioning.md` for the registry schema and legacy inventory. The old format-1 checkpoint remains loadable for ordinary continuation but predates question conditioning; retrain before using `--question`.

## Speech-corpus continuation run

The first Roll Call speech run used 1,200 target-only events selected with at least 1,000 Trump words and an 80% target-word share. It trained one epoch on 10.11 million model tokens with a 7.26M-parameter model. The whole-date chronological split was 954/122/124 train/validation/test events (1988-06-05 through 2024-10-25, 2024-10-26 through 2026-01-06, and 2026-01-13 through 2026-11-02). It produced 251,841/30,896/32,627 continuation chunks and zero Q&A chunks. Validation continuation loss was 4.3586; held-out test loss was 4.4586 with top-1 accuracy 0.2747. On the identical held-out target tokens, a train-frequency unigram baseline with add-one smoothing scored loss 6.0024 and top-1 accuracy 0.0668. These are preliminary, uncalibrated continuation results from one run, not response prediction or semantic behavior accuracy. The ignored checkpoint is `checkpoints/rollcall-long-speech.pt`.

## Historical results and limits

The GovInfo corpus contains mixed presidential documents, not a manually verified speech set. Its current data and heuristic Q&A labels do not support representative quality claims. Model scores are uncalibrated.

The earlier format-1 experiment used a 6.78M-parameter model for five epochs on the then-current corpus and reached test token loss 4.2168 and top-1 accuracy 0.3320. Samples were repetitive and sometimes reproduced editorial notes. Those historical metrics do not evaluate question conditioning or external evidence.

The first format-2 run used the v2 paragraph linker for one epoch. On its held-out paragraph-based Q&A windows, combined transcript+question scored token loss/top-1 4.3369/0.2431, transcript-only 3.9996/0.2574, and question-only 4.8784/0.1869. It showed no question-conditioning benefit. These are legacy plumbing checks, not timestamped event-level results; the local checkpoint is ignored and must not be used as a live-response claim.
