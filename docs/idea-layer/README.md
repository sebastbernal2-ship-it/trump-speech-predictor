# Active specification: context → speech ideas

This is the canonical project specification. Root `README.md` and `CONTEXT.md` summarize it.

## Objective

For a historical Trump speech, predict the ideas he will address from news/context available strictly before the event. The active headline classifier uses the **seven days before the event** as a frozen comparison baseline, not as an established optimal horizon. A separate story-graph research track will preserve older active stories, updates, and entity connections, then learn or validate their relevance using chronological data. No live or upcoming event is required.

Each event is a supervised example. For the frozen baseline:

```text
X7(event) = pre-event headline titles from [event_date - 7 days, event_date)
y(event)  = ideas expressed in the target-speaker transcript
```

The experimental story-graph representation may include older source observations that remain relevant through a story's development, but every observation/relation must be available before the event cutoff. It will be evaluated separately against the same outcome definition.

The transcript is the realized outcome, not a feature. For event `i`, never use its transcript, extracted labels, post-event material, or later context to make its prediction. Earlier examples' paired inputs and outcome labels may train the model for later examples. **Earlier speech text, historical phrase frequency, and speaker-history features are not predictor inputs.**

## Active code and command

- Model/evaluation: `speechprefix/stages/ideas/model.py`
- Shared target extraction: `speechprefix/headline_idea_matching.py`
- CLI: `context-idea-classifier-evaluate` (evaluate/tune) and `context-idea-predict` (rank ideas for a strictly later event from a fitted artifact)
- Tests: `tests/test_forecasting_context_to_ideas.py`

Run the frozen historical context experiment with a new output path:

```sh
uv run python -m speechprefix context-idea-classifier-evaluate \
  --snapshots data/rollcall-context-snapshots-retro-combined-v1.jsonl \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --output data/rollcall-context-to-idea-evaluation-NEW.json
```

The classifier uses headline-title unigram/bigram features and a Bernoulli Naive Bayes multi-label model. It selects smoothing, feature-frequency settings, a cap on the most recent deduplicated headline titles, and logit temperature by validation binary log loss. Publisher-name suffixes are stripped; displayed prediction labels use the most frequent training-speech surface form for each phrase ID. Evaluation reports binary log loss (including explicit penalties for labels unseen in earlier training), ranking quality, candidate coverage, and per-event predictions. A no-context issue-prevalence reference is reported separately and is not an input to the model. The evaluation artifact contains a final refit and is accepted by `context-idea-predict`, which rejects target dates at or before the training cutoff and verifies artifact hashes.

## Available historical data and current result

The current frozen context union contains 20 topic-family snapshots plus 18 separately collected news snapshots (38 unique events total). Six have no target phrases under the current extractor; 32 are usable. The whole-date chronological split is 25 training, 3 validation, and 4 development events. A 72-setting validation search selected `alpha=4.0`, `min_df=3`, 20 most recent unique titles, stable capture-order ties, and temperature `8.0`.

On the four development events, the stable-order recent-headline model scored binary log loss **0.708**, nDCG@10 **0.475**, P@4 **0.563**, and recall@10 **0.371**. The no-context prevalence reference scored **0.562**, **0.439**, **0.563**, and **0.315** respectively. Of 45 extracted target IDs, 16 appear in the top 10, three are unavailable in prior training, and 26 known IDs rank below top 10. The selected model's three-event validation ranking was very poor (nDCG@10 **0.101**). The development ranking lift is unstable, while binary loss is worse than baseline. See [`../idea-error-analysis.md`](../idea-error-analysis.md) for per-event diagnosis and prioritized next steps.

All 38 snapshots were retrieved after their target speeches. Feed publication dates earlier than an event do not establish that the stories or search results were available then. The development sample is tiny; the target labels are incomplete lexical phrase extractions. These figures are retrospective tuning diagnostics, not semantic accuracy or independent test performance.

Artifacts are local and Git-ignored:

- `data/rollcall-context-snapshots-retro-combined-v1.jsonl` — SHA-256 `96e94152e3b9318dbe8ab34dc28fd0d3a8f8718fa6f0ed5316cb143d43f33314`
- Earlier pre-loss-correction report: `data/rollcall-context-to-idea-evaluation-v3.json` — SHA-256 `4f8c44f012ba34c005ffb59f27a99c2ba29391deb760150f95081e9e27accb7d`
- Corrected-loss all-headline report/model: `data/rollcall-context-to-idea-evaluation-v4.json` — SHA-256 `e5bdb26713fe8ec8ecec885da89599d41aa3c0a80f1f25b75114a4a120c9ab7c`
- Superseded recent-headline report with date-tie ordering bug: `data/rollcall-context-to-idea-evaluation-v5.json` — SHA-256 `fccae70aa7610f1b4dd7f4f0f3dafeee93b0ca10ad410cfb5ae2efcc1408acf0`
- Current stable-order model and selection audit: `data/rollcall-context-to-idea-evaluation-v7.json` — SHA-256 `5589eb462e6db386a972393822192b36f449ed88b5668d930442821391b5fd87`

## Story-level context and graph research

The next input-representation experiment is to preserve individual articles while modeling exact duplicates, syndication, updates, related coverage, and time-bounded entity-profile claims as a heterogeneous graph. Its design and strict as-of boundary are in [`story-graph-architecture.md`](story-graph-architecture.md). The graph contract currently validates provenance, temporal fields, hashes, and cutoff projection; story clustering, learned weighting, entity-profile ingestion, and Q-GNN remain unimplemented. Do not change the active seven-day model until a separately versioned chronological comparison is complete.

## Data/review rules

- Keep event dates and source publication dates explicit. Same-day date-only headlines are excluded.
- Do not use headline summaries, locations, event titles, or metadata unless explicitly included in a separately versioned feature set and timestamped as available; the current classifier uses headline titles only.
- Preserve raw evidence and immutable run outputs; never overwrite local snapshots or evaluation artifacts.
- The phrase extractor is a weak target-label mechanism. It is not exhaustive or semantic gold. Preserve misses and uncertainty; do not claim accuracy beyond the specified label task.
- Transcript interpretation or semantic matching is AI-only. Assistant-reviewed labels are references, not objective truth. See [`ai-review-policy.md`](ai-review-policy.md).
- Additional input-source availability and leakage audit (including the Twitter archive and the UCSB crawler restriction): [`input-source-audit.md`](input-source-audit.md). A separate 2019 local-source backtest is documented in [`2019-full-year-preliminary-study.md`](2019-full-year-preliminary-study.md); it is diagnostic and does not update the active headline-only classifier.

## Explicitly out of scope

Speech-history recurrence as a prediction feature, incoming/live speech collection, speech wording or plan generation, Q&A, and provider-prompt generation are outside the active objective. Historical prototype implementations remain outside the active forecasting path and are not dependencies of this model.
