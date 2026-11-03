# SpeechPrefix

## Primary direction: an event-specific idea forecast

The project is being reorganized around one standalone layer: **before a scheduled speech, rank the ideas Trump is likely to address using current, time-verified news, his recent public posts, and details known about that event.** As the speech unfolds, this layer may update its idea ranking from turns already observed. It does not generate wording or a whole speech.

The current-events idea system is **not implemented end to end yet**: source validation, a local evidence ledger, pre-cutoff snapshots, news/post import paths, proposal validation, and an evidence-weighted ranking baseline exist. No Trump-post archive or semantic candidate generator is connected, and no prospective forecast or accuracy result exists. The source/history rankers and speech-generation experiments below are earlier prototypes, not evidence that this goal has been achieved. See [`docs/idea-layer/README.md`](docs/idea-layer/README.md) for the scope, evidence rules, repository boundary, and next steps.

This is a research project, not a promise to reproduce a future speech. Ranking scores are not probabilities unless calibrated; all forecasts must preserve source citations and an exact prediction cutoff. No live audio/transcript feed is currently included.

## Existing prototypes (separate from the primary direction)

The repository also contains **SpeechPlan-H**, a from-scratch hierarchical-code planner joined to a GPT-style autoregressive decoder by cross-attention, plus n-gram, standalone GPT-2/NCM, continuation, history-only issue, and retrospective context experiments. These remain separate baselines/prototypes and are not dependencies of the standalone idea layer.

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
python3 -m speechprefix train --input data/speeches.jsonl --output checkpoints/ngram-run-001.json
python3 -m speechprefix predict --model checkpoints/ngram-run-001.json --prefix "We are going to"
python3 -m speechprefix stream --model checkpoints/ngram-run-001.json
python3 -m speechprefix evaluate --input data/speeches.jsonl
python3 -m speechprefix forecast --input data/speeches.jsonl
python3 -m speechprefix audit --input data/speeches.jsonl
.venv/bin/python -m speechprefix segment --input data/speeches.jsonl
python3 -m speechprefix turn-import --transcript data/page.txt --metadata data/event.json --output data/events.jsonl
.venv/bin/python -m speechprefix context-fetch --as-of 2026-09-30 --query "artificial intelligence"
```

The GovInfo importer reads `GOVINFO_API_KEY` from the current environment or a repo-root `.env` file. It downloads at most 25 speech-like CPD documents per run by default, only for Trump's presidential periods, and appends them to ignored `data/speeches.jsonl`. Use `--period 2017` or `--period 2025` to target a term. Increase `--limit` deliberately after checking the first batch. The filter is title-based and will miss some remarks and include some non-speech records; review the resulting records and applicable reuse terms before training.

In stream mode, submit each new chunk on its own line. A blank line exits. For SpeechPlan-H, train and generate locally:

```sh
.venv/bin/python -m speechprefix hybrid-train --input data/speeches.jsonl --output checkpoints/speechplan-h-run-001.pt --epochs 1
.venv/bin/python -m speechprefix hybrid --model checkpoints/speechplan-h-run-001.pt --prefix "We have been discussing our plans" --question "What is the plan?"
```

For a real-transcript weak-label experiment, create the keyword plans and train with the explicit weak-label policy. For semantic plan training, use only reviewed plans in `data/rollcall-speech-plans.jsonl`:

```sh
.venv/bin/python -m speechprefix weak-plan-labels
.venv/bin/python -m speechprefix explicit-plan-train --events data/rollcall-events.jsonl --plans data/rollcall-weak-speech-plans.jsonl --label-policy weak-extractive --output checkpoints/explicit-plan-weak-run-001.pt
.venv/bin/python -m speechprefix explicit-plan-predict --model checkpoints/explicit-plan-weak-run-001.pt --prefix "Observed speech prefix"
.venv/bin/python -m speechprefix explicit-plan-evaluate --model checkpoints/explicit-plan-weak-run-001.pt --plans data/rollcall-weak-speech-plans.jsonl --device cpu --cpu-threads 4
```

For human-reviewed semantic plans, pass `--plans data/rollcall-speech-plans.jsonl` and keep the default `--label-policy reviewed`. The older `pretrained-hybrid` command is only a candidate-fusion comparison baseline; it may use external checkpoints. All trained-model commands append immutable run metadata and artifact hashes to `models/registry.jsonl` and refuse to overwrite an existing artifact. SpeechPlan-H's architecture and known limits are described in `docs/speechplan-hybrid.md`; see `docs/speechplan-h-flowchart.md` for the visual breakdown.

`segment` is the legacy paragraph-based pipeline; its question-response links are heuristic and are not the target representation for live exchanges. With source permission, `import-rollcall` collects Roll Call Factba.se's timestamped event pages into the separate, ignored `data/rollcall-events.jsonl`; `split-rollcall` derives disjoint Trump speech, Q&A-candidate, other-Trump, and non-Trump files without changing the master corpus. `turn-import` parses a manually saved time-coded page. Both preserve speaker turns, timestamps, page annotations, and source attribution without guessing which question a response answers. `prepare-rollcall-speeches` exports long, Trump-dominant speech text for continuation training; timestamped Q&A events still need reviewed response links before supervised use. See `data/README.md` for the split and export details. Legacy paragraph-based Q&A results are not live-response performance. `context-fetch` calls the public Federal Register API without an API key and stores dated titles/abstracts locally; it excludes documents published on the prediction date because transcript event times are unavailable. The current Federal Register sample is unpaired and is not model input. Current legacy-corpus counts and limitations are in `docs/data-audit.md`.

## Prediction tasks, separate evaluations

- **Standalone idea layer (primary direction):** combine time-verified current events, Trump’s attributable recent posts, and pre-event details to rank likely speech ideas; append evidence-grounded updates as visible transcript turns arrive. The package boundary is `speechprefix/idea_layer/`. Its current code validates provenance, captures Google News headlines, normalizes permitted post archives, freezes inputs, and ranks cited proposals with a transparent baseline; no post archive or semantic proposal generator is connected, and evaluation/live updates remain to be built. No prospective sample or reliable accuracy result exists. See `docs/idea-layer/README.md`.
- **Legacy pre-speech baselines:** the history ranker predicts recurring extracted phrases; the retrospective context tools retrieve source-supported phrases. They do not establish that Trump will discuss a current event. Existing snapshot audits found no strict prospective sample. Treat their scores as exploratory weak-label diagnostics, not evidence of the primary idea layer’s accuracy; see `docs/context-theme-forecast.md` and `docs/evaluation.md`.
- **Continuation:** Given an observed transcript prefix, predict its next word(s). Replay held-out speeches sequentially and score each prediction before revealing the next word.
- **In-speech plan path:** Given the same kind of observed prefix, predict one theme → section → talking-point path. Keep weak pseudo-label diagnostics separate from semantic idea metrics; human-reviewed plans are required for semantic accuracy.
- **Question-anchored response:** Given the observed turns and a reviewer-selected question/prompt trigger in a time-coded event, predict the target speaker's likely ideas and exact words. Triggers can include follow-ups and the target speaker's own clarifying questions. Score ideas separately from wording and track whether another speaker responds or the target does not.

Never mix these scores. A continuation model sees the target transcript prefix; a pre-speech model must not. Question-response evaluation must reveal only turns observable by its prediction time.

SpeechPlan-H is a from-scratch prototype with a coarse-to-fine residual VQ planner whose predicted codes become cross-attention memory for an autoregressive Transformer decoder. It is a small word-level language model, not a pretrained large language model. Its training code performs a chronological, whole-date split, but the legacy input corpus contains mixed presidential records that have not been manually verified as speeches. Treat those metrics as plumbing checks. SpeechPlan-H can train continuation from the derived long-speech export, but its response-training path is not connected to timestamped Q&A events. The separate explicit plan-path model predicts a theme → section → talking-point path from a transcript prefix. Its `weak-extractive` policy can train on real transcripts using clearly marked source-topic and keyword pseudo-labels; those results measure heuristic reproduction, not verified semantic idea accuracy. Human-reviewed plans are still required for semantic claims. The model does not generate wording or condition the SpeechPlan-H decoder; see `docs/hierarchical-speech-planner.md`.

## Data and evaluation cautions

Use transcripts you have permission to use. The GovInfo importer is an initial source adapter for the CPD collection, not a complete archive of Trump's speeches. It does not capture campaign rallies that were not published in CPD. Confirm the applicable reuse terms for each record. Record source, date, event, and transcript-quality notes for every speech. Keep whole speeches together in chronological train/validation/test splits. Do not randomly split overlapping excerpts. Report log loss and top-k accuracy for continuation; assess probability calibration before describing scores as confidence. Pre-speech forecasts need their own dated target labels and metrics.

See `docs/architecture.md` for component boundaries, `docs/evaluation.md` for the current protocol, `docs/context-theme-forecast.md` for the context-to-theme experiment, `docs/speechplan-hybrid.md` for the SpeechPlan-H architecture and training procedure, and `docs/model-versioning.md` for the append-only artifact registry and reproducibility metadata.
