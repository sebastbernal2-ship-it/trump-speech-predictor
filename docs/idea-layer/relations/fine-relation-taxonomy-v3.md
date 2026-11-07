# Fine-relation taxonomy v3: stricter textual warrant

**Status:** frozen after the 17-event taxonomy-v2 diagnostic failed its fine-edge review screen. V3 has **not** been applied to that sample or any evaluation sample. Its synthetic examples are protocol fixtures, not labels from the corpus.

## Why revise

On the 2021 sample, the v2 generator emitted 298 edges, 228 typed as `supports`. A same-provider diagnostic reviewer considered only 165/257 reviewed edges plausible, with 51 uncertain and 41 mismatched. This does not establish error rates, but it raises a clear operational risk: relation labels may have inferred rhetorical function from topical co-occurrence. V3 therefore requires explicit textual warrant and abstention rather than making subtle distinctions from relevance alone.

## Rules

Check the separate `expresses` anchor first; do not add a fine edge for the same mention/idea pair. For any proposed fine edge:

- **`supports`** — The mention states a reason, premise, or evidence connection for the idea. Require wording that makes the reason/evidence role explicit (for example, “because”, “therefore”, or an unmistakable argument). A relevant or persuasive fact alone is insufficient.
- **`illustrates`** — The mention is explicitly or unmistakably framed as a concrete instance of the idea. A fact that could serve as an example is not automatically an illustration.
- **`elaborates`** — The mention clearly specifies what the idea includes: scope, mechanism, or implementation. Same-topic detail without a clear connection is insufficient.
- **`qualifies`** — The mention explicitly limits, conditions, corrects, or excepts the idea. Mere additional detail is not a qualification.
- **`contextualizes`** — The mention is explicitly framed as setting/background for interpreting the idea, while making no stronger relation. This is not a fallback class for relevant co-occurrence.
- **Abstain** — Omit the edge if the text does not establish the connection, or if two relation types remain plausible. An omitted edge means “not proposed,” not “unrelated.”

The ordered taxonomy constants (`FINE_RELATION_DEFINITIONS`, `FINE_RELATION_DECISION_RULES`, and `FINE_RELATION_BOUNDARY_EXAMPLES`) in `speechprefix/idea_layer/relation_edges.py` carry version `fine-relation-taxonomy-v3`. Both extractor and diagnostic reviewer receive the same payload. New artifacts record the taxonomy version; engine/prompt versions are incremented. Validation here remains structural, never semantic.

## Synthetic contrast set

All examples use the idea **“Extend evening bus service.”** Their relation is warranted only by the stated wording:

| Mention | Type | Textual warrant |
|---|---|---|
| “Because the last bus leaves at 9:45 and hospital shifts end after 10, extend evening service.” | `supports` | “Because” directly states the reason. Remove that argument link and the annotator must abstain. |
| “For instance, a nurse leaving the hospital at 10:30 has no bus and walks four miles home.” | `illustrates` | “For instance” explicitly frames a concrete case as an example. |
| “The proposal includes hourly buses on routes 2 and 6 until midnight.” | `elaborates` | “The proposal includes” explicitly attaches non-limiting implementation detail. |
| “The extension would apply only on weekends.” | `qualifies` | “Only” explicitly narrows scope. |
| “Against this backdrop, the city added two hospital districts as development spread westward.” | `contextualizes` | “Against this backdrop” explicitly marks setting; without the contextual frame, do not infer the relation. |

## Evaluation boundary

The v2 diagnostic and frozen sample are not to be rerun or tuned. V3 requires a newly frozen sample, the same explicit structural/error accounting, and provider-independent diagnostics if that provider can be configured. Same-provider agreement is not a substitute. A passing model-as-judge threshold still would not provide semantic gold; human review or independent reference labels would be needed to certify accuracy. Keep relation edges out of prediction features.
