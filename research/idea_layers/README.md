# Speech-content research layers

Research code is organized by the semantic object or task it produces. This is separate from `speechprefix/idea_layer/`, which contains reusable runtime/model components.

| Folder | Role | Data status |
|---|---|---|
| `outcomes/` | Transcript-only extraction of quote-grounded mentions, point/idea nodes, links, audits, and retries. Mentions and points remain distinct but are generated/validated as one linked outcome artifact. | Target-speech outcome labels only; never same-event features. |
| `points/` | Talking-point classification and independent human-reference packet generation. | Candidate annotations/references; not gold until independently adjudicated. |
| `topics/` | Broader, non-destructive grouping of existing point IDs. | Topic outcome candidates; no point merges and no active predictor use. |
| `relations/` | Typed edges among mentions/points and expression anchors. | Separate diagnostic layer; failed quality gates remain explicit. |
| `stories/` | Real-world story identity, news/post circulation, and event-centered retrieval audits. | Potential input-side evidence only when time-bounded and proven available by cutoff; retrospective case studies are not live features. |
| `prediction/` | Context-to-candidate generation experiments. | Only eligible pre-event context may be input; target transcripts/labels remain excluded. |
| `legacy_atoms/` | Earlier transcript-atom extractor, audit, augmentation, and retry tools. | Historical prototype; distinct from the current linked mention/point outcome contract. |

## Artifact layout

New local/generated artifacts should go under Git-ignored `data/source-archives/speech-content/<same-layer>/`, for example:

```text
data/source-archives/speech-content/
  outcomes/
  points/
  topics/
  relations/
  stories/
  predictions/
  legacy_atoms/
  reference_annotations/
```

Keep raw/source transcripts separate from model-generated labels, and keep human-reference forms separate from candidate inventories. Existing frozen artifacts are not moved or rewritten during this organization pass; this includes the active v2 human-reference packet bundle. New runs should specify a layer-specific `--output` path. Preserve manifests, hashes, source timestamps, provider/model/prompt versions, and explicit error states.

## Prediction boundary

A target speech's mentions, points, topics, stories derived from its transcript, and all such labels are outcomes. Different layers can be separate forecast targets. Features for a target event must be generated from independently eligible, strictly pre-event sources. See [`docs/idea-layer/prediction/multilevel-prediction-architecture.md`](../../docs/idea-layer/prediction/multilevel-prediction-architecture.md). The frozen seven-day classifier remains unchanged; every new level or feature family requires separate chronological evaluation.
