# Active architecture

## Context-to-ideas supervised pipeline

```text
historical event context snapshot
  └─ headline titles published in [event_date - 7 days, event_date)
       └─ headline unigram/bigram features
            └─ supervised multi-label issue classifier
                 └─ ranked idea IDs + model scores
                      └─ compare against target-speaker transcript idea labels
```

Training examples are chronological pairs `(pre-event context, realized transcript ideas)`. For any scored event, fit only on examples from strictly earlier event dates. Group all events from the same date so they do not teach each other's predictions. The target transcript is outcome-only for its own event. It can be training supervision for later events.

Active implementation:

- `speechprefix/stages/ideas/model.py`: feature construction, Bernoulli Naive Bayes fitting/prediction, validation search, and chronological backtest. The evaluator serializes the all-pairs final refit in its immutable JSON report. `predict_context_ideas` and the `context-idea-predict` CLI apply it only to event dates strictly later than its training cutoff, and verify both report/model hashes.
- `speechprefix/headline_idea_matching.py`: target-speaker filtering and current compact issue-phrase labels.
- `speechprefix/cli.py`: `context-idea-classifier-evaluate` and `context-idea-predict` commands.
- `tests/test_forecasting_context_to_ideas.py`: synthetic cutoff, chronological split, and immutable-output checks.

## Other stage boundaries

The remaining stage boundaries are scaffolded so future work has a designated home:

- [`speechprefix/stages/ideas/README.md`](../speechprefix/stages/ideas/README.md): active home for the supervised idea predictor.
- [`speechprefix/stages/sentences/README.md`](../speechprefix/stages/sentences/README.md): reserved for a future sentence-level model. It is intentionally empty of model code until its contract and evaluation are defined.
- [`speechprefix/pipelines/README.md`](../speechprefix/pipelines/README.md): reserved for orchestration across stages; no end-to-end pipeline exists yet.

These are explicit scaffolds, not claims that later stages already exist. The rest of `speechprefix/` still contains legacy or unrelated prototypes; they are not active-model dependencies or feature sources. See [`idea-layer/README.md`](idea-layer/README.md) for the task contract and [`evaluation.md`](evaluation.md) for the current backtest.
