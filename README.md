# Trump Speech Idea Forecaster

**Project objective:** predict ideas in a Trump speech from news and context available before it, then compare predictions with the realized transcript. The active headline classifier uses the **seven days before the speech** as a frozen baseline; a separate story-graph research track will test whether older stories, updates, and entity links improve forecasts without assuming a fixed horizon. Develop and tune offline against historical data. No live or upcoming speech is required.

The frozen benchmark's input is that event's preceding-week context; experimental story-graph inputs may draw on older pre-cutoff evidence when a story remains active or develops. In every variant, the target transcript is the realized output used to create training labels and measure error; it is **never a feature for predicting that same event**. Earlier examples' context/label pairs can train the input-to-output mapping. Prior speech text, phrase frequency, and speaker-history features are not target-event inputs.

## Current implementation

- `speechprefix/stages/ideas/model.py`: supervised Bernoulli headline-to-idea classifier. Features are title unigrams/bigrams from the seven days before an event; labels are compact issue phrases extracted from its target-speaker transcript.
- CLI: `context-idea-classifier-evaluate` tunes/backtests; `context-idea-predict` applies its serialized final fit to a strictly later context snapshot. Validation selects smoothing, feature frequency, and logit temperature by binary log loss; reported scores remain uncalibrated.
- Current historical input: 38 unique retrospective snapshots; 32 produce extractable target labels. The stable-order recent-headline variant modestly improves nDCG@10 (0.475 vs 0.439) and recall@10 (0.371 vs 0.315) versus the no-context reference, ties P@4, and loses on binary loss (0.708 vs 0.562) on only four development events. It is runnable, but **not a demonstrated successful forecaster**.
- News snapshots were retrieved after their target speeches. Their feed dates are historical evidence for a retrospective experiment, not proof of point-in-time availability. The speech idea labels are also incomplete lexical proxies, not semantic ground truth.

Full task definition and current experiment: [`docs/idea-layer/README.md`](docs/idea-layer/README.md). Story-level graph design and Q-GNN-compatible data contract: [`docs/idea-layer/story-graph-architecture.md`](docs/idea-layer/story-graph-architecture.md). Data caveats: [`docs/data-audit.md`](docs/data-audit.md). AI-only transcript review policy: [`docs/idea-layer/ai-review-policy.md`](docs/idea-layer/ai-review-policy.md).

## Run the historical experiment

```sh
uv sync --extra test
uv run python -m speechprefix context-idea-classifier-evaluate \
  --snapshots data/rollcall-context-snapshots-retro-combined-v1.jsonl \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --output data/rollcall-context-to-idea-evaluation-NEW.json
```

Reports and local collected corpora must remain Git-ignored and immutable; use a new output name for every run. Frozen input and model-report hashes are recorded in [`docs/idea-layer/README.md`](docs/idea-layer/README.md). To apply the fitted model to a strictly later snapshot, run `uv run python -m speechprefix context-idea-predict --model data/rollcall-context-to-idea-evaluation-v7.json --snapshot <later-snapshot.json> --output data/idea-prediction-NEW.json`. The predictor rejects same/earlier event dates and same-day/future headlines.

## Repository map

- `speechprefix/stages/ideas/`: active supervised context-to-ideas model, evaluation, and prediction API.
- `speechprefix/stages/sentences/`: reserved future-stage scaffold; no model implementation yet.
- `speechprefix/pipelines/`: reserved for future orchestration; no end-to-end pipeline exists yet.
- `speechprefix/`: shared transcript parsing, issue-target extraction, CLI, and legacy research prototypes.
- `tests/`: unit and synthetic-fixture tests.
- `data/`: local ignored transcripts, news snapshots, predictions, and evaluations.
- `docs/`: active specification, architecture, data audit, evaluation contract, and review policy.

Superseded prototype documentation has been removed. Historical prototype code remains outside the active forecasting path; it is not imported by the active model.
