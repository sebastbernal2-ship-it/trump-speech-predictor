# Retrospective context-to-ideas evaluation protocol

This is the active evaluation contract for `context-idea-classifier-evaluate`.

## Example and leakage boundary

The frozen active baseline uses headline titles published from seven days before the event through the day before it. The new story-graph experiment is not restricted to that window: it may use the retained pre-cutoff archive, with story persistence, updates, and continuous time features represented explicitly. The target remains the compact issue-phrase set extracted from the target-speaker transcript. That transcript and all target-derived data are outcome-only for that event. Earlier examples' context/label pairs may train later predictions. No earlier transcript text, prior issue-frequency feature, or speaker-history score enters the target event's feature vector.

## Story graph and temporal-weight evaluation

- Keep every article/source observation; represent exact duplicates, syndication, updates, and related coverage as distinct typed relations. Do not treat duplicate outlet copies as independent corroboration or collapse substantive updates into duplicates.
- For each chronological fold, construct or slice the graph using only source evidence and relation/profile facts available before that event's cutoff. Do not let future articles retroactively define an earlier story cluster. Record graph-builder/entity-resolution versions and their training cutoffs.
- Compare the unchanged seven-day item-level baseline against deduplicated story-level and update-aware representations over broader pre-cutoff context. Do not assume a seven-/fourteen-day limit, source weight, or recency half-life is correct. Tune temporal attention/aggregation on validation folds only; report its selected parameters and sensitivity.
- Report raw-item → exact-duplicate → syndication/update-candidate → story/development reductions; publisher diversity, repeated-copy volume, update cadence, context-age distributions, and per-event predictions/misses. Preserve cluster uncertainty and source citations.
- Without human review, exact URL/content identities and synthetic tests can diagnose mechanical behavior, but do not call automatically generated semantic story clusters gold labels.

## Chronological training and evaluation

- Sort by event date; group all events on a date.
- Use earliest 80% of dates for training, following 10% for validation, final 10% as retrospective development scoring. Report exact IDs and date ranges.
- Tune only on validation binary log loss, breaking ties by nDCG@10. Fit features/vocabulary on earlier examples only.
- For each development target, fit from examples with strictly earlier event dates. As targets become historical for later dates, their input/label pairs may enter later fits.
- Report binary log loss, P@4, recall@10, nDCG@10, target candidate coverage, per-event misses, and a no-context prevalence reference. Scores are not probabilities unless calibrated.
- After selection, fit the model on all eligible historical input/output pairs. Keep this final fit distinct from out-of-fold scores.

## Interpretation

The current set contains only 38 unique retrospective context snapshots and 32 usable labeled events, with four development events. Snapshot retrieval happened after the target speech; the date-filtered article publication fields do not prove point-in-time availability. The target extractor is lexically incomplete. Scores therefore describe a small retrospective development experiment against weak labels, not semantic truth or independent generalization. Preserve and report failures. AI transcript review, if used, follows [`ai-review-policy.md`](ai-review-policy.md) and is not objective gold.
