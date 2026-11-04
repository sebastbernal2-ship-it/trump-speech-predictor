# Active specification: context → speech ideas

This is the canonical project specification. Root `README.md` and `CONTEXT.md` summarize it.

## Objective

For a historical Trump speech, predict the ideas he will address from news/context headlines published during the **seven days before that event**. Train and tune offline on the available historical input/output pairs. No live or upcoming event is required.

Each event is a supervised example:

```text
X(event) = pre-event context headlines from [event_date - 7 days, event_date)
y(event) = ideas expressed in the target-speaker transcript
```

The transcript is the realized outcome, not a feature. For event `i`, never use its transcript, extracted labels, post-event material, or later context to make its prediction. Earlier examples' paired inputs and outcome labels may train the model for later examples. **Earlier speech text, historical phrase frequency, and speaker-history features are not predictor inputs.**

## Active code and command

- Model/evaluation: `speechprefix/forecasting/context_to_ideas.py`
- Shared target extraction: `speechprefix/headline_idea_matching.py`
- CLI: `context-idea-classifier-evaluate`
- Tests: `tests/test_forecasting_context_to_ideas.py`

Run the frozen historical context experiment with a new output path:

```sh
uv run python -m speechprefix context-idea-classifier-evaluate \
  --snapshots data/rollcall-context-snapshots-retro-combined-v1.jsonl \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --output data/rollcall-context-to-idea-evaluation-NEW.json
```

The classifier uses headline-title unigram/bigram features and a Bernoulli Naive Bayes multi-label model. It selects smoothing and feature-frequency settings by validation binary log loss. Evaluation reports binary log loss, ranking quality, candidate coverage, and per-event predictions. A no-context issue-prevalence reference is reported separately and is not an input to the model.

## Available historical data and current result

The current frozen context union contains 20 topic-family snapshots plus 18 separately collected news snapshots (38 unique events total). Six have no target phrases under the current extractor; 32 are usable. The fixed chronological split is 24 training, 4 validation, and 4 development events. The latest run selected `alpha=2.0`, `min_df=1`.

On the four development events, the headline classifier scored binary log loss **3.357**, nDCG@10 **0.428**, P@4 **0.500**, and recall@10 **0.315**. The no-context prevalence reference scored **0.324**, **0.439**, **0.563**, and **0.315** respectively. The classifier is overconfident and does not beat this simple reference. This is a negative first result, not proof that context cannot work.

All 38 snapshots were retrieved after their target speeches. Feed publication dates earlier than an event do not establish that the stories or search results were available then. The development sample is tiny; the target labels are incomplete lexical phrase extractions. These figures are retrospective tuning diagnostics, not semantic accuracy or independent test performance.

Artifacts are local and Git-ignored:

- `data/rollcall-context-snapshots-retro-combined-v1.jsonl` — SHA-256 `96e94152e3b9318dbe8ab34dc28fd0d3a8f8718fa6f0ed5316cb143d43f33314`
- `data/rollcall-context-to-idea-evaluation-v3.json` — SHA-256 `4f8c44f012ba34c005ffb59f27a99c2ba29391deb760150f95081e9e27accb7d`

## Data/review rules

- Keep event dates and source publication dates explicit. Same-day date-only headlines are excluded.
- Do not use headline summaries, locations, event titles, or metadata unless explicitly included in a separately versioned feature set and timestamped as available; the current classifier uses headline titles only.
- Preserve raw evidence and immutable run outputs; never overwrite local snapshots or evaluation artifacts.
- The phrase extractor is a weak target-label mechanism. It is not exhaustive or semantic gold. Preserve misses and uncertainty; do not claim accuracy beyond the specified label task.
- Transcript interpretation or semantic matching is AI-only. Assistant-reviewed labels are references, not objective truth. See [`ai-review-policy.md`](ai-review-policy.md).

## Explicitly out of scope

Speech-history recurrence as a prediction feature, incoming/live speech collection, speech wording or plan generation, Q&A, and provider-prompt generation are outside the active objective. Historical prototype implementations remain outside the active forecasting path and are not dependencies of this model.
