# Fine-relation taxonomy v2

**Status:** frozen definitions used on the 17-event 2021 diagnostic sample. This taxonomy is a versioned prompt/annotation protocol, not a validated semantic ontology. It was not used on the failed 2020 v1 sample. Examples below are synthetic and were not taken from the corpus or audit outputs. The 2021 test failed the fine-edge review thresholds; a stricter taxonomy v3 was frozen afterward and has not yet been evaluated.

## Scope

These labels connect a transcript-local mention to an already-anchored higher-level idea. They do not replace the separate direct-expression `expresses` anchor. Check direct expression first: if the mention itself states the idea, use an anchor and do not emit a fine edge for that same pair.

Emit at most one fine relation per mention/idea pair. An omitted edge means “no sufficiently clear fine relation proposed,” not “the mention is unrelated.” If the evidence fits multiple types and the distinction cannot be resolved by the rules below, abstain.

## Operational definitions and decision order

Apply the tests in order after checking for a direct-expression anchor:

1. **`qualifies`** — The mention changes the idea's scope or truth conditions by limiting it, conditioning it, correcting it, or stating an exception. A non-limiting detail is not enough.
2. **`illustrates`** — The mention presents a particular case as an instance of the idea: “here is the idea happening in this case.” Prefer this when an identifiable case is offered as an example, even if it is also persuasive evidence.
3. **`supports`** — The mention supplies a reason, premise, or evidence for believing or acting on the idea. The relation should be paraphrasable as “this is a reason for the idea.” Prefer general evidence/reasons over a concrete case framed as an example.
4. **`elaborates`** — The mention adds general scope, mechanism, or implementation detail that clarifies what the idea includes without limiting or correcting it. It should be paraphrasable as “more specifically, the idea means/includes this.”
5. **`contextualizes`** — The mention supplies relevant setting/background that helps interpret the idea but does not itself give a reason, instantiate the idea, specify its content, or alter its scope.
6. **Abstain** — Use no fine edge when no relation is clear or when two labels remain plausible after applying the distinctions above.

The order is a tie-breaker, not permission to infer an unstated relation. For example, a limitation takes `qualifies` only when it actually narrows/corrects the target idea; otherwise a detail may `elaborate` it. A background fact is `contextualizes` only when the text does not use it as a reason for the idea.

## Generic contrast set

All examples use the same abstract idea, **“Extend evening bus service.”** They are boundary illustrations, not evidence about any real city.

| Mention | Label | Why this relation fits | Nearby label to reject |
|---|---|---|---|
| “The last bus leaves at 9:45, although many hospital shifts end after 10.” | `supports` | General evidence/reason for extending service. | Not `illustrates`: no particular person/case is being presented as an instance. |
| “A nurse leaving the hospital at 10:30 has no bus and walks four miles home.” | `illustrates` | A concrete case instantiates the service gap. | Not `supports` merely because the case is persuasive; choose `supports` for a general reason/evidence statement. If the text does not make the example-vs-evidence distinction clear, abstain. |
| “The proposal would run hourly buses on routes 2 and 6 until midnight.” | `elaborates` | Adds general route/schedule implementation details. | Not `qualifies`: it does not limit or correct a broader claim. |
| “The extension would apply only on weekends.” | `qualifies` | “Only” narrows the proposal's scope. | Not `elaborates`: the condition changes which cases the broader idea covers. |
| “The city added two hospital districts as development spread westward.” | `contextualizes` | Relevant setting, with no stated reason or specification of the bus proposal. | If the mention explicitly argues that the growth makes evening service necessary, use `supports` instead. |

A direct-expression boundary example: **“We should extend evening bus service.”** directly states the idea, so it is an `expresses` anchor and not one of the fine-edge labels.

## Implementation and provenance

The versioned definitions, ordered decision rules, and synthetic examples are in `speechprefix/idea_layer/relation_edges.py` as `FINE_RELATION_TAXONOMY_VERSION`, `FINE_RELATION_DEFINITIONS`, `FINE_RELATION_DECISION_RULES`, and `FINE_RELATION_BOUNDARY_EXAMPLES`. Both the fine-edge extractor and its AI-only diagnostic reviewer receive the same taxonomy payload. New relation artifacts record the taxonomy version; extractor/audit prompt versions are incremented accordingly.

Taxonomy v2 was applied to the separately frozen 2021 sample documented in [`relation-label-quality-bar-v1.md`](relation-label-quality-bar-v1.md). Its audit did not meet the fine-edge consistency thresholds. Do not use the failed 2020 sample to tune taxonomy v2. The stricter v3 protocol is a response to that diagnostic, not a result; it must be tested on a newly frozen sample and cannot establish semantic accuracy without independent reference labels or human review. Keep relation candidates out of prediction features.
