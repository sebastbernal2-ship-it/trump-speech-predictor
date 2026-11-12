# AI-only transcript review policy

This is the active policy when the project creates assistant-reviewed semantic references or adjudicates model ideas against a transcript. It does not change the supervised training setup: the target transcript remains the realized outcome/label for its own event, never that event's predictor input.

## Requirements

- The assistant is the sole reviewer. Do not request, await, or imply human annotation or adjudication.
- Read the complete available target-speaker transcript before writing a semantic outcome inventory. Record transcript coverage, source/transcript hashes, reviewer/model provenance, rationale, and uncertainty.
- Cite exact transcript turn IDs for every idea. Separate substantive ideas from ceremonial mentions, duplicate surface labels, and ambiguous/incomplete evidence.
- Freeze transcript-only outcome references before comparing them with model predictions when the workflow allows. Record any nonblind exposure explicitly; post-hoc reviews are development diagnostics.
- Match prediction candidates after the reference inventory is fixed. Use `match`, `partial_match`, `not_supported`, or `uncertain`; explain boundaries and cite target turns. Do not force ambiguity into a negative.
- Preserve immutable source and review artifacts. Corrections create a new version; do not rewrite previous references.

## Claims

Assistant-reviewed ideas are **interpretive references**, not objective semantic ground truth and not human-reviewed labels. Report agreement with the assistant-reviewed references, not semantic accuracy. Weak phrase-extractor metrics describe only agreement with that extractor. Do not claim human validation, inter-rater reliability, or objective truth.

The annotated 2026 examples and older development reviews are nonblind/inspected development material. They may diagnose model errors and support retrospective tuning, but selected scores are not independent test performance.
