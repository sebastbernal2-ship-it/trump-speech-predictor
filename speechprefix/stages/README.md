# Model stages by output granularity

These folders represent **different prediction/generation layers**, from semantic planning to surface form. Keep each layer independently developable, testable, evaluable, and versioned.

```text
pre-event features ──> topics ──> ideas/points ──> sentences ──> words ──> characters
```

The arrows describe a possible hierarchical pipeline, not mandatory conditioning: topic and idea forecasts can also be predicted directly from pre-event context and must be benchmarked separately. The current active seven-day model predicts ideas directly. No topic→idea, sentence, word, or character model is active.

## Stage ownership

| Stage | Output unit | Status | Folder |
|---|---|---|---|
| Topics | Broad issue/theme addressed | Reserved; topic outcome contract and independent references required | `topics/` |
| Ideas | Specific talking points/claims/proposals | Active pre-event idea classifier | `ideas/` |
| Sentences | Sentence-level realization/selection | Scaffold only | `sentences/` |
| Words | Versioned lexical/token sequence | Scaffold only | `words/` |
| Characters | Character-level surface sequence | Scaffold only; define Unicode unit/normalization first | `characters/` |

Each stage owns its input/output contract, model/training implementation, artifact versioning, and unit-specific evaluation. A stage consumes a frozen, hashed upstream prediction artifact—not another stage's private code or mutable in-memory state. Keep each stage's tests under the matching `tests/stages/<stage>/` directory.

## Data/leakage contract

A target speech's topic, idea, sentence, word, and character annotations are all **outcomes**. They can supervise separate chronological prediction tasks, but never become same-event input features. For a hierarchical inference run, downstream stages consume the **predicted** upstream artifact; teacher-forced results using gold upstream labels must be reported separately from end-to-end results.

Story/event graphs and source provenance are an orthogonal **input-evidence axis**, not another text-generation granularity. The transcript mention/point/topic annotation tools under `research/idea_layers/` create outcome labels; they are separate from these model stages.

The reserved stages remain empty of model code until their target units, contracts, cutoff rules, and independent metrics are specified. See [`tests/stages/README.md`](../../tests/stages/README.md) for test ownership and [`docs/architecture.md`](../../docs/architecture.md) for the active model/data flow.
