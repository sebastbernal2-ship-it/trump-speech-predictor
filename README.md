# SpeechPrefix

An offline research prototype for two separate tasks: forecasting likely content before a speech, and updating next-word predictions as a transcript prefix arrives.

This is an experiment, not a promise to reproduce a future speech. Models return candidate probabilities estimated from training data; treat them as uncalibrated until evaluation says otherwise. No live audio or transcript feed is included.

## Current prototype

The project now includes **SpeechPlan-H**, a from-scratch hierarchical-code planner joined to a GPT-style autoregressive decoder by cross-attention. It does not load GPT-2 or Stanford NCM weights. It is trained locally on the chronological training split. N-gram, standalone GPT-2/NCM, and the earlier candidate-fusion path are separate comparison baselines. Pre-speech phrase forecasting remains a separate task.

## Requirements

Python 3.10 or newer. The baseline has no third-party dependencies. Optional local model packages are installed in `.venv` for this checkout; activate with `source .venv/bin/activate` or use `.venv/bin/python` directly.

```sh
.venv/bin/python -m pip install -e '.[ml]'
```

The `ml` extra installs PyTorch and Transformers; SpeechPlan-H uses those libraries but downloads no model weights. The separate `llm` and `ncm-generate` comparison commands use cached external checkpoints and may download assets when first run. The older `pretrained-hybrid` command also uses them; it is not SpeechPlan-H.

## Quick start

For the continuation baseline, put permitted transcript text in `data/transcripts.txt`, with speeches separated by blank lines. For the pre-speech baseline, use dated JSONL records as described in `data/README.md`. Then:

```sh
python3 -m speechprefix import-govinfo --limit 25
python3 -m speechprefix train --input data/speeches.jsonl --output model.json
python3 -m speechprefix predict --model model.json --prefix "We are going to"
python3 -m speechprefix stream --model model.json
python3 -m speechprefix evaluate --input data/speeches.jsonl
python3 -m speechprefix forecast --input data/speeches.jsonl
python3 -m speechprefix audit --input data/speeches.jsonl
.venv/bin/python -m speechprefix segment --input data/speeches.jsonl
.venv/bin/python -m speechprefix context-fetch --as-of 2026-09-30 --query "artificial intelligence"
```

The GovInfo importer reads `GOVINFO_API_KEY` from the current environment or a repo-root `.env` file. It downloads at most 25 speech-like CPD documents per run by default, only for Trump's presidential periods, and appends them to ignored `data/speeches.jsonl`. Use `--period 2017` or `--period 2025` to target a term. Increase `--limit` deliberately after checking the first batch. The filter is title-based and will miss some remarks and include some non-speech records; review the resulting records and applicable reuse terms before training.

In stream mode, submit each new chunk on its own line. A blank line exits. For SpeechPlan-H, train and generate locally:

```sh
.venv/bin/python -m speechprefix hybrid-train --input data/speeches.jsonl --output checkpoints/speechplan-h.pt --epochs 1
.venv/bin/python -m speechprefix hybrid --model checkpoints/speechplan-h.pt --prefix "We have been discussing our plans" --question "What is the plan?"
```

The older `pretrained-hybrid` command is only a candidate-fusion comparison baseline; it may use external checkpoints. SpeechPlan-H's architecture and known limits are described in `docs/speechplan-hybrid.md`; see `docs/speechplan-h-flowchart.md` for the visual breakdown.

`segment` splits transcript paragraphs, distinguishes explicit questions from reporter floor prompts, and links likely president responses in transcript order. Those links are heuristic; unmarked text stays unattributed. SpeechPlan-H trains separate transcript and question channels and reports Q&A results apart from ordinary continuation. Add `--question` at generation time after training a new checkpoint. `context-fetch` calls the public Federal Register API without an API key and stores dated titles/abstracts locally; it excludes documents published on the prediction date because transcript event times are unavailable. External evidence is accepted only when the checkpoint was trained with dated, question-linked evidence pairs; see the JSONL schema in `CONTEXT.md`. The current Federal Register sample is unpaired, so it is not yet model input. Current counts and limitations are in `docs/data-audit.md`.

## Two tasks, two evaluations

- **Continuation:** Given an observed prefix from a speech, predict its next word(s). Evaluation replays held-out speeches sequentially and scores each prediction before revealing the next word.
- **Pre-speech forecast:** Given only historical speeches, rank recurring phrases likely to appear in the next speech. The initial baseline uses dated JSONL metadata and walk-forward evaluation. This is a phrase-recurrence baseline, not a semantic topic model; topic labels and event-aware inputs remain future work.

Never mix these scores. A continuation model sees the opening of the target speech; a pre-speech model must not.

SpeechPlan-H is a from-scratch prototype with a coarse-to-fine residual VQ planner whose predicted codes become cross-attention memory for an autoregressive Transformer decoder. It is a small word-level language model, not a pretrained large language model. Its training code performs a chronological, whole-date split, but the current corpus contains mixed presidential records that have not been manually verified as speeches. Treat all resulting metrics as plumbing checks until a clean speech set is assembled. Exact wording and semantic segment quality need separate evaluations.

## Data and evaluation cautions

Use transcripts you have permission to use. The GovInfo importer is an initial source adapter for the CPD collection, not a complete archive of Trump's speeches. It does not capture campaign rallies that were not published in CPD. Confirm the applicable reuse terms for each record. Record source, date, event, and transcript-quality notes for every speech. Keep whole speeches together in chronological train/validation/test splits. Do not randomly split overlapping excerpts. Report log loss and top-k accuracy for continuation; assess probability calibration before describing scores as confidence. Pre-speech forecasts need their own dated target labels and metrics.

See `docs/architecture.md` for component boundaries, `docs/evaluation.md` for the current protocol, and `docs/speechplan-hybrid.md` for the SpeechPlan-H architecture and training procedure.
