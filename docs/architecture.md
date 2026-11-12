# Active architecture

## Frozen seven-day context-to-ideas baseline

```text
historical event context snapshot
  └─ headline titles published in [event_date - 7 days, event_date)
       └─ headline unigram/bigram features
            └─ supervised multi-label issue classifier
                 └─ ranked idea IDs + model scores
                      └─ compare against target-speaker transcript idea labels
```

This remains the frozen benchmark, not a justified optimal time horizon.

## Experimental story-graph input representation

```text
source evidence ledger (publication + observed/available time + provenance)
  └─ article observations (keep every source edition)
       ├─ exact duplicate / syndication relations
       ├─ update, correction, and related-story relations with confidence
       └─ entity mentions + dated, cited entity-profile claims
            └─ strict as-of graph projection at target cutoff
                 └─ future: candidate-conditioned typed message passing / Q-GNN
                      └─ ranked ideas with story paths and source citations
```

`story_graph.py` currently validates typed nodes/edges, source-ledger references, temporal validity, content hashes, and strict or explicitly retrospective as-of projections. Clustering, update detection, entity-profile ingestion, adaptive horizon/relevance learning, and Q-GNN are not implemented. See [`idea-layer/story-graph-architecture.md`](idea-layer/stories/story-graph-architecture.md) for the contract and phase boundaries.

Training examples are chronological pairs `(pre-event context, realized transcript ideas)`. For any scored event, fit only on examples from strictly earlier event dates. Group all events from the same date so they do not teach each other's predictions. The target transcript is outcome-only for its own event. It can be training supervision for later events.

Active implementation:

- `speechprefix/stages/ideas/model.py`: feature construction, Bernoulli Naive Bayes fitting/prediction, validation search, and chronological backtest. The evaluator serializes the all-pairs final refit in its immutable JSON report. `predict_context_ideas` and the `context-idea-predict` CLI apply it only to event dates strictly later than its training cutoff, and verify both report/model hashes.
- `speechprefix/headline_idea_matching.py`: target-speaker filtering and current compact issue-phrase labels.
- `speechprefix/cli.py`: `context-idea-classifier-evaluate` and `context-idea-predict` commands.
- `tests/stages/ideas/test_ideas_stage.py`: synthetic cutoff, chronological split, and immutable-output checks.

## Independent output-granularity stages

The stage hierarchy and folder ownership are documented in [`speechprefix/stages/README.md`](../speechprefix/stages/README.md). Each stage has its own development and test directory for topic, idea, sentence, word, or character outputs. Story/source evidence is an orthogonal input axis, not a text-output stage.

The remaining stage boundaries are scaffolded so future work has a designated home:

- [`speechprefix/stages/ideas/README.md`](../speechprefix/stages/ideas/README.md): active home for the supervised idea predictor.
- [`speechprefix/stages/topics/README.md`](../speechprefix/stages/topics/README.md): reserved for a separate broad-topic predictor.
- [`speechprefix/stages/sentences/README.md`](../speechprefix/stages/sentences/README.md): reserved for a future sentence-level model. It is intentionally empty of model code until its contract and evaluation are defined.
- [`speechprefix/stages/words/README.md`](../speechprefix/stages/words/README.md) and [`speechprefix/stages/characters/README.md`](../speechprefix/stages/characters/README.md): reserved for separate word/token and literal-character models, with their own unit contracts and metrics.
- [`speechprefix/pipelines/README.md`](../speechprefix/pipelines/README.md): reserved for orchestration across stages; no end-to-end pipeline exists yet.

These are explicit scaffolds, not claims that later stages already exist. The rest of `speechprefix/` still contains legacy or unrelated prototypes; they are not active-model dependencies or feature sources. See [`idea-layer/README.md`](idea-layer/README.md) for the task contract and [`evaluation.md`](evaluation.md) for the current backtest.
