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

- `speechprefix/forecasting/context_to_ideas.py`: feature construction, Bernoulli Naive Bayes fitting/prediction, validation search, and chronological backtest. The evaluator reports a hash of a final refit; it does not serialize a deployable model.
- `speechprefix/headline_idea_matching.py`: target-speaker filtering and current compact issue-phrase labels.
- `speechprefix/cli.py`: `context-idea-classifier-evaluate` command.
- `tests/test_forecasting_context_to_ideas.py`: synthetic cutoff, chronological split, and immutable-output checks.

The rest of `speechprefix/` contains legacy or unrelated prototypes. They are not dependencies or feature sources for the active predictor. See [`idea-layer/README.md`](idea-layer/README.md) for exact scoring definitions and [`evaluation.md`](evaluation.md) for the backtest contract.
