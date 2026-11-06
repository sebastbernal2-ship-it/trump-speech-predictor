# Two-stage expression anchors and fine-grained relation edges

**Status:** bounded research prototype; not a validated label pipeline. This implements the separation between direct-expression anchors and optional fine-grained relations. The fine edges remain candidate annotations, not active forecasting features or graph ground truth.

## Design

The earlier layered extractor asked one model pass to produce `expresses`, `supports`, `illustrates`, `elaborates`, `qualifies`, and `contextualizes` links together. The schema required at least one `expresses` link per idea, but did not establish that all direct-expression pairs had been found. In the 14-event transcript-only audit, relation-type concerns included cases where a direct expression appeared to be categorized as contextual or supporting evidence. This prototype separates those jobs:

1. **Expression-anchor pass:** inspect a frozen mention/idea inventory and transcript, and identify mention→idea pairs where the mention itself directly expresses the idea. It can explicitly abstain on an idea by listing it as unanchored. It does not see legacy fine-relation labels.
2. **Fine-relation pass:** receive the separate, source-bound anchor artifact and frozen nodes. It can only propose optional non-`expresses` relations between mentions and ideas whose ideas have direct anchors. It cannot revise anchors. Ambiguous relationships should be omitted rather than forced.
3. **Model-as-judge diagnostic:** review supplied anchors and fine edges separately, and optionally suggest missing anchor pairs. These verdicts/suggestions are not semantic gold or a completeness estimate.

The code is deliberately scoped to transcript mention→idea relationships. It is not yet a general relation extractor for stories, organizations, claims, market states, or other heterogeneous graph nodes. The existing news/story graph contracts remain separate.

## Contracts and implementation

- `speechprefix/idea_layer/semantic_anchors.py` defines the source-bound anchor artifact. It binds the exact outcome-row and transcript hashes, deterministic anchor IDs, an anchor-set hash, and the partition between anchored and explicitly unanchored ideas. Each anchor resolves through a mention to its existing exact-turn citations. Validation establishes structure/provenance only; it does not establish entailment or completeness.
- `speechprefix/idea_layer/relation_edges.py` defines optional fine edges. `expresses` is not legal in this layer. A fine edge must resolve to a source mention and an anchored idea; it cannot duplicate an anchor or assign multiple fine relation labels to one mention/idea pair. Proposed IDs/pairs are checked before record construction. The artifact records its anchor-artifact hash and source-outcome hash. No confidence score is emitted.
- `research/extract_layered_expression_anchors.py` is the bounded anchor-only engine. `research/extract_layered_relation_edges.py` is the separate fine-edge engine and requires explicit event IDs; it never defaults to processing the corpus. Invalid edges are rejected before output; successful repairs retain attempt counts and prior validation errors, and exhausted retries produce a source-bound `model_error` row instead of omitting the event or implying a successful empty edge set. The auditor reviews anchors on such rows but marks fine-edge review as not run. `research/audit_layered_relation_edges.py` performs the separate AI-only diagnostic.
- The relation layer inherits its exact evidence through its mention IDs and the source outcome's quote/turn provenance. No new quote is generated for an edge.
- The event transcript remains an outcome-label source only. None of these artifacts or labels were used as same-event prediction inputs; forecastability was not evaluated by these passes.

## Bounded pilot

The pilot uses three already-inspected 2019 development events whose v2.3 outcomes had been extracted by `deepseek / deepseek-v4-pro`: `dt45-19-ed9fee63` (2019-08-07), `dt45-19-18ebde99` (2019-10-30), and `dt45-19-1143aee4` (2019-11-07). They are a targeted development sample, not a random or untouched evaluation set.

### Anchor pass diagnostics

The original one-pass outcomes contained 53 direct-expression pairs over these events. A separate `openai-codex / gpt-6.1-sol` anchor-only pass produced 70 pairs: it retained 52 of the 53 original pairs, omitted one, and added 18 new pairs. A second anchor-only inventory from `openai-codex / gpt-6-luna` produced 97 pairs. On these three cases, the second inventory included all 70 pairs from the first and added 27 more. The `gpt-6.1-sol` set had no explicitly unanchored ideas; this does **not** prove all direct expressions were found.

A different alias, `openai-codex / gpt-6-astra`, judged the first pass's 70 anchors `direct` (70/70) but also suggested 18 additional anchor pairs. Seventeen of those suggestions appeared in the second model's 27 additions; ten of the second model's additions were not suggested by the auditor, and one auditor suggestion was absent from the second inventory. This overlap is useful disagreement/coverage diagnostic evidence, not truth: all three models were served by the same provider, aliases are mutable, and no independent reference inventory exists. In particular, the second model's 97-pair inventory has not itself been audited here.

### Fine-relation diagnostics

Using the first separate anchor artifact, `openai-codex / gpt-6-luna` proposed 36 optional fine edges: 17 `contextualizes`, 12 `elaborates`, 5 `supports`, 1 `illustrates`, and 1 `qualifies`. One event received no fine-edge proposals. A separate model alias, `openai-codex / gpt-6-astra`, reviewed those 36 edges: 27 `plausible`, 8 `uncertain`, and 1 `mismatch`. Counts by relation were:

| Relation | Plausible | Uncertain | Mismatch |
|---|---:|---:|---:|
| `contextualizes` | 14 | 3 | 0 |
| `elaborates` | 11 | 1 | 0 |
| `illustrates` | 0 | 1 | 0 |
| `qualifies` | 0 | 0 | 1 |
| `supports` | 2 | 3 | 0 |

These are model verdicts, **not** 27/36 accuracy or precision. The reviewer shared the extraction provider and cannot provide an independent truth check. The small sample and omitted-edge problem prevent any recall estimate. The high uncertainty among `supports` edges and the single `qualifies` mismatch reinforce that the relation distinctions still need operational sharpening; they do not show that the relation types themselves are unnecessary.

Per-event results:

| Event ID | Date | Anchor pairs (first pass) | AI-suggested additional anchors | Fine edges | Edge verdicts (plausible / uncertain / mismatch) |
|---|---|---:|---:|---:|---|
| `dt45-19-ed9fee63` | 2019-08-07 | 17 | 1 | 0 | 0 / 0 / 0 |
| `dt45-19-18ebde99` | 2019-10-30 | 22 | 4 | 8 | 6 / 1 / 1 |
| `dt45-19-1143aee4` | 2019-11-07 | 31 | 13 | 28 | 21 / 7 / 0 |

The zero-edge case is abstention by the relation engine; it is not evidence that the transcript contains no meaningful relationships. Similarly, the auditor's proposed missing anchors are candidates for further analysis, not verified omissions.

## Provenance, privacy, and reproducibility

The three input outcome rows are in the Git-ignored local artifact `data/source-archives/2019-layered-outcomes-v2.3-complete.jsonl` (SHA-256 `d466cd4b903a9dc9e5e2789632d549a9466628379c8e06a0b6b433f26877dd52`). The two anchor inventories, fine-edge candidates, and model-as-judge output are also local and Git-ignored:

- First anchor inventory: `data/source-archives/2019-layered-expression-anchors-v1.jsonl` — SHA-256 `b0f062259649952772b5f0f86cc836502a089d9c00f1fddd81bc062ea89fff51`
- Second anchor inventory (`gpt-6-luna`): `data/source-archives/2019-layered-expression-anchors-v2-gpt6-luna.jsonl` — SHA-256 `33c7977a6dfd536f8afa1b205d83c57bec48180ab0f839f03317ca15627d5a07`
- Fine-edge candidates: `data/source-archives/2019-layered-fine-relation-edge-candidates-v2.jsonl` — SHA-256 `70562e92040c692ee94342a576f0350b69adaf34a60e9d30c6ebbbad5b7e8480`
- Anchor/edge audit: `data/source-archives/2019-layered-relationship-audit-v2.jsonl` — SHA-256 `bb768fcb777a01ae2568d77ebf78a00807136f509d957e99f81feb89db5d1aeb`

The output manifests record source hashes, model/prompt versions, and per-event results. Full target-speaker transcripts and transcript-derived outcome layers for these cases were sent to the configured `openai-codex` provider for the anchor passes, relation pass, and audit. The provider/model aliases are mutable. No transcript text, quotes, candidate edges, audit outputs, or checkpoints are committed.

## Conclusion and next gate

The separation is technically useful: a direct-expression anchor is now a distinct, hash-bound artifact, while fine relations are optional and cannot mutate it. But this pilot does **not** establish that the labels are “good enough.” The 18 model-suggested anchor omissions and the second inventory's additional 27 links show that anchor completeness is not settled; the relation review still has substantial uncertainty. No result here certifies semantic correctness, edge recall, cross-market graph utility, or forecastability.

A subsequent, frozen 12-event 2020 test applied explicit anchor-agreement and relation-review thresholds. It failed the per-event anchor agreement screen; two relation extractions failed structural validation; fine-edge review also fell below the specified thresholds. Cross-provider review was blocked by provider balance/configuration. See [`relation-label-quality-bar-v1.md`](relation-label-quality-bar-v1.md) for the predeclared bar, per-event sample, hashes, and full result. The current status remains **candidate-only**.

Before using these edges as graph features:

1. Keep the failed 2020 sample frozen; do not retry/tune on those events. The two extraction validation failures and anchor disagreements remain diagnostic evidence.
2. Use the now-frozen [`fine-relation-taxonomy-v2.md`](fine-relation-taxonomy-v2.md) on a **newly selected** sample only. Its contrastive examples are synthetic; they have not been scored or used to relabel the failed sample.
3. Obtain a configured, funded second provider for the required independent model diagnostic; provider diversity still does not create semantic gold.
4. Keep candidate relation edges out of the forecasting model. The active seven-day baseline remains unchanged.
