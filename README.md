# Trump Speech Idea Forecaster

**Project objective:** predict the ideas in a Trump speech from the news and event context available during the **seven days before that speech**, then compare predictions with the ideas in the realized transcript. Develop and tune offline against the historical corpus. No live or upcoming speech is required.

The model input for a target event is that event's preceding-week context. Its transcript is the realized output used to create training labels and measure error; it is **never a feature for predicting that same event**. Earlier examples' context/label pairs can train the input-to-output mapping. Prior speech text, phrase frequency, and speaker-history features are not target-event inputs.

## Current implementation

- `speechprefix/forecasting/context_to_ideas.py`: supervised Bernoulli headline-to-idea classifier. Features are title unigrams/bigrams from the seven days before an event; labels are compact issue phrases extracted from its target-speaker transcript.
- CLI: `context-idea-classifier-evaluate`. It uses chronological event-date splits, chooses smoothing settings by validation binary log loss, and reports log loss, ranking metrics, candidate coverage, and per-event predictions.
- Current historical input: 38 unique retrospective snapshots (20 topic-family snapshots and 18 news snapshots); 32 produce extractable target labels. The four-event development slice did **not** beat the no-context prevalence reference and showed severe overconfidence. This is a weak initial baseline, not a successful model.
- News snapshots were retrieved after their target speeches. Their feed dates are historical evidence for a retrospective experiment, not proof of point-in-time availability. The speech idea labels are also incomplete lexical proxies, not semantic ground truth.

Full task definition and current experiment: [`docs/idea-layer/README.md`](docs/idea-layer/README.md). Data caveats: [`docs/data-audit.md`](docs/data-audit.md). AI-only transcript review policy: [`docs/idea-layer/ai-review-policy.md`](docs/idea-layer/ai-review-policy.md).

## Run the historical experiment

```sh
uv sync --extra test
uv run python -m speechprefix context-idea-classifier-evaluate \
  --snapshots data/rollcall-context-snapshots-retro-combined-v1.jsonl \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --output data/rollcall-context-to-idea-evaluation-NEW.json
```

Reports and local collected corpora must remain Git-ignored and immutable; use a new output name for every run. Frozen input and report hashes are recorded in [`docs/idea-layer/README.md`](docs/idea-layer/README.md).

## Repository map

- `speechprefix/forecasting/`: active context-to-ideas model and training/evaluation logic.
- `speechprefix/`: shared transcript parsing, issue-target extraction, CLI, and legacy research prototypes.
- `tests/`: unit and synthetic-fixture tests.
- `data/`: local ignored transcripts, news snapshots, predictions, and evaluations.
- `docs/`: active specification, architecture, data audit, evaluation contract, and review policy.

Superseded prototype documentation has been removed. Historical prototype code remains outside the active forecasting path; it is not imported by the active model.
