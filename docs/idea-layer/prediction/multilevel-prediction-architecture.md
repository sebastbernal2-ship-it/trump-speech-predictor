# Multi-level speech-content prediction architecture

**Status:** research design. This extends the transcript-outcome and topic-layer prototypes; it does not change the active model or frozen seven-day baseline.

## Objective

Predict speech content at multiple semantic resolutions and analyze how evidence propagates between levels:

1. **Topic/theme:** broad issue or conversation thread.
2. **Talking point:** a specific claim, position, proposal, appeal, or event-specific focus (`ideas` in the current transcript-outcome schema).
3. **Story/event:** a particular real-world referent linked to one or more points. Story identity is separate from point identity.
4. **Transcript mention/expression:** exact post-event wording/evidence. This is normally provenance for the outcome, not the forecast target; phrase-level prediction would be a separate task.

These levels form a linked graph, not a flat label list. A topic groups points; a story may be referenced by multiple points; a point can concern multiple stories. Relatedness, evidence, and entailment are typed links, not merge instructions.

## Critical separation: outcomes vs. as-of features

A target speech's transcript, mentions, points, topics, story links, and any labels derived from them are **outcome-only**. They may define the labels for topic-level and point-level tasks, train on prior completed speeches, and score predictions. They must never be same-event input features.

Input-side features must be constructed independently from sources that were available by the target event cutoff, such as dated news articles and eligible speaker/public posts. Those sources can be mapped to input-side topic/point candidates and story evidence, with timestamps, hashes, retrieval/availability limitations, and source type retained. Do not copy the target speech's topic or point labels into its context feature bundle. Features extracted from historical speeches may be used only under an explicitly versioned policy proving those speeches precede the cutoff; speech-history recurrence remains a separate ablation, not an assumed feature.

Thus each semantic level can be a **prediction target**, and corresponding pre-event signals can be candidate **features**, but target-event outcome labels themselves are not features.

## Task definitions

Keep each task separate and compare chronologically:

- **Topic task:** predict/rank the topics addressed by the target speech from pre-event information.
- **Point task:** predict/rank the specific talking points. Evaluate semantic matching and granularity, not only phrase overlap.
- **Story task:** retrieve or rank the real-world stories/events that support the target points. Do not treat story retrieval as proof that a point was spoken.
- **Hierarchical/two-stage task:** predict topics first, then points conditional on predicted topics; compare against direct flat point prediction and topic probabilities aggregated from point predictions.
- **Expression task (optional):** predict direct speaker wording only if separately justified and evaluated. Exact transcript mentions remain outcome evidence in the present contract.

For every task, report its target definition, candidate-generation policy, matching rules, cutoff, feature set, and failure/abstention state. An extractor's output is not semantic gold; use independently annotated references before interpreting forecast performance.

## Features and analysis

Candidate input features may be aligned across levels, while remaining as-of safe:

- **Topic-level:** dated coverage volume/diversity, prominence, trajectory/onset/bursts, and source-type composition, all computed from eligible pre-cutoff sources.
- **Point-level:** evidence that a specific proposition is being discussed, speaker statements/posts, supporting or opposing arguments, and the breadth/strength of matched source evidence.
- **Story-level:** event occurrence date, article/story clusters, syndication/update relations, and evidence coverage; retain source-level provenance and uncertainty.
- **Cross-level:** topic exposure aggregated from its member stories/points; point evidence nested under topics; story-to-point warrant links. Do not propagate scores across a link unless the feature rule says how and is separately ablated.

Analyze at minimum: direct topic vs. direct point prediction; point predictions aggregated to topics vs. directly predicted topics; topic-then-point vs. flat point prediction; evidence/source ablations; performance by speech length/type and topic frequency; calibration and coverage; unresolved topic/point matches; and whether topic-level gains conceal poor point specificity.

## Evaluation and safety gates

1. Freeze and manually review point boundaries before using transcript labels as semantic evaluation targets. Review topic memberships separately from point boundaries.
2. Construct multiple independent annotator references before exposing candidate outputs; adjudicate and preserve unresolved matches. The three-speech pilot is diagnostic, not a test corpus.
3. Keep the frozen seven-day headline classifier unchanged. Compare every level and feature family in separately versioned, chronological runs against that baseline.
4. Keep all same-event transcript-derived values out of inputs; verify cutoff projection programmatically. Historical publication/retrieval dates do not by themselves prove real-time availability.
5. Preserve source provenance, raw evidence references, hashes, model/prompt versions, error states, and source limitations. Keep generated labels/transcripts local and Git-ignored.

The current topic grouper is only a non-destructive partition over candidate point IDs. Its one-speech output demonstrates the representation, not semantic accuracy. No multi-level model, topic feature bundle, or baseline change has been implemented.
