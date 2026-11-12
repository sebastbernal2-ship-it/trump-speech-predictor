# Human reference protocol: speech conversation points (v2)

**Status:** frozen pilot instructions. This protocol and sample are for transcript-outcome annotation only. Transcript content and annotations must never be used as same-event prediction inputs. The reference sample is diagnostic, not a representative corpus or an extractor pass/fail benchmark.

## Objective

Independently record the distinct key conversation points the target speaker makes in the supplied transcript. We are evaluating point coverage, specificity, atomicity, evidence faithfulness, and salience—not whether the speaker's claims are true or forecastable.

## Independent annotation procedure

1. Two annotators work **independently**. Do not discuss the speech or compare notes until both have submitted and frozen their files.
2. Read the entire transcript packet in order before writing the inventory. Annotate only `[TARGET]` turns; `[CONTEXT_*]` turns are provided to clarify responses and references. Use only the supplied packet; do not consult web sources, existing model-generated plans/ideas, previous audits, news results, or the other annotator.
3. Add one row per distinct key point in the corresponding blank CSV. Quote exact evidence from `[TARGET]` turns and identify packet-local turn IDs. Multiple independent points may share a turn/quote.
4. Re-read all turns and check for omissions and over-splitting. Record genuine uncertainty rather than resolving it from outside knowledge.
5. Do not inspect extractor candidates until both independent inventories have been timestamped/frozen. Then adjudicate by discussion, recording whether each disagreement is a miss, unsupported point, merge, split, overly broad label, salience disagreement, or evidence/link issue. Preserve unresolved cases.

## Unit and boundaries

A **key conversation point** is one specific proposition, argument, position, proposal, appeal, or event-specific focus that the target speaker substantively addresses: a main/developed/repeated point or a discrete claim/proposal given clear independent emphasis. It must state what the speaker says (subject + claim/action/position), not just name a topic, person, country, program, or event.

- **Split** two points when each is independently emphasized/developed and could be discussed or forecast independently. Shared topic, actor, event, paragraph, or political frame does not justify merging. A single story/event may be referenced by distinct points.
- **Do not split** every statistic, example, name, subordinate policy detail, or rhetorical aside into its own point. Keep it as supporting context unless the target speaker independently develops it as a focus.
- Distinguish a claim about an event from the event itself when the speech gives them independent emphasis. Do not infer that two statements are the same idea merely because one mentions the other.
- Attribute disputed statements to the speaker (“the speaker claims/argues…”). Annotation is not fact-checking.
- Preserve independent event-specific points even if they seem unlikely to be forecast. Forecastability is out of scope for this annotation.

Generic vs. specific: `immigration` is not a point; `the speaker argues that proposal X would increase border crossings` is point-shaped **only if that proposition is actually expressed and emphasized in the transcript**. Do not use the example to infer content.

## CSV fields

- `annotator_point_id`: your local stable ID, such as `P001`.
- `point_summary`: one concise, neutral, proposition-specific sentence.
- `kind`: `policy`, `political`, `factual_claim`, `campaign_appeal`, `event_specific`, `personal_or_ceremonial`, `other`, or `unclear`.
- `speaker_relation`: `claims_or_argues`, `supports_or_endorses`, `opposes`, `proposes_or_requests`, `forecasts`, `describes_or_focuses_on`, or `unclear`.
- `prominence`: `central`, `repeated_or_developed`, `secondary`, or `brief_but_independent`. Do not use prominence to omit a distinct key point.
- `evidence_turn_ids`: one or more packet-local IDs separated by `;`. Evidence must be from `[TARGET]` turns; context turns are not evidence of the target's point.
- `exact_quote`: short exact quote(s) copied from the target speaker's turn(s), separated by ` | ` if needed. Retain case and punctuation; no reconstructed sentence.
- `broader_point_id`: optional ID only if another independently listed point genuinely subsumes this one (specific-to-general). Shared topic is not enough.
- `relation_to_broader_point`: if used, state `specific_to`; otherwise blank. Evidence/support/example relations should be noted in `boundary_note`, not used to merge points.
- `boundary_note`: brief rationale, alternative reading, or unresolved merge/split issue. Leave blank if none.

The CSV is blank except for its header. Add rows; do not edit the transcript packet. You may use additional rows as needed; there is no target point count.

## Agreement and adjudication (not yet performed)

Keep independent inventories unchanged. After both are frozen, compare proposed points through evidence-backed one-to-one matches and explicitly adjudicate one-to-many and many-to-one cases as split/merge disagreements. Report at least:

- point-level precision/recall/F1 under adjudicated semantic matching (with unresolved/ambiguous matches separately reported);
- unsupported-point rate and transcript-evidence validity;
- under-specific and over-broad summaries;
- merge/split error counts;
- recall weighted by the adjudicated prominence level, alongside unweighted recall;
- point-boundary agreement before adjudication and agreement on kind/prominence separately.

A two-annotator pilot on three speeches has wide uncertainty and does not establish general semantic accuracy. Do not tune on these three and then call the same sample a test set. Freeze further development speeches chronologically before reporting any forecasting result. No transcript-derived label may enter same-event prediction features.

## Frozen packet sample and provenance

The three packets were selected from the local 2025 Roll Call transcript collection across target-speaker length and event-format strata. The sample IDs were checked against the local layered idea-inventory and model-output artifacts included in the freeze audit; their contents were not consulted. This is a blind-reference procedure, not a claim that no other model outputs exist anywhere. Lexington is excluded because its prior generated inventory and provisional breakdown were exposed in the conversation. The exact event mapping, source-record/transcript hashes, packet hashes, model-artifact scan, and selection rationale are in the Git-ignored freeze manifest generated by `research/idea_layers/points/create_keypoint_reference_packets.py` under `data/source-archives/keypoint-reference-sample-v2/`. The packet files contain transcript text and must remain local/Git-ignored.
