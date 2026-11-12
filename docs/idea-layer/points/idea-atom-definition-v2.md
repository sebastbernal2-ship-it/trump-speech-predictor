# Speech conversation-point definition v2

**Status:** revised extraction protocol, not semantic gold. Versioned separately from v1; prior annotations are unchanged and must not be described as v2 outputs. The v2.1 refinement adds a key-point salience threshold after the first v2 pilot over-split subordinate details.

## Unit

A **conversation point** (candidate idea) is one specific, key proposition, argument, position, proposal, appeal, or event-specific subject that the speaker substantively addresses. It is a main/developed/repeated point or a discrete claim/proposal given clear independent emphasis—not every sentence-level assertion. It must be summarized as what the speaker actually says—not just a topic, person, policy domain, or event name. The summary should identify the subject and the speaker's expressed claim, stance, or requested action, with no outside knowledge or inferred motive.

Examples of insufficient labels: “impeachment,” “Ukraine,” “Biden,” “the economy,” “foreign policy.” A speech may contain multiple distinct points under each label.

## Atomicity

Split two key points when each is independently emphasized/developed and could be discussed, accepted, rejected, or forecast independently. Shared names, paragraph, topic, event, or political framing do not justify merging. A single event can ground multiple separate claims; a single political argument can cite several events as evidence. Keep each point distinct and express cross-references as links, not a merged summary.

Do not split every fact, number, name, example, or repetition into an idea. Keep details as transcript mentions when they support or illustrate the same proposition. Do not promote every subordinate policy component or rhetorical aside just because it could be debated separately. Promote a detail to an idea only if the speaker substantively develops it as a distinct focus. Preserve genuinely independent event-specific points even if they are not forecastable.

## Required extraction workflow

1. Read transcript turns in order, restricted to the target speaker.
2. Record substantive transcript-local mentions with exact quote and turn ID.
3. Form one specific idea for each independently meaningful key conversation point. Do not target a fixed number or merge points to reduce output size.
4. Link each idea to a mention that directly expresses it; link supporting details separately.
5. Reread all turns and audit for omitted points, broad topic-only summaries, accidental merges, and over-splitting of supporting detail.
6. Keep confidence/forecastability separate from coverage: uncertain forecastability is not grounds to omit a point.

## Quality gates and limits

Keep these dimensions separate: quote integrity, coverage/omissions, claim specificity, atomicity, mention-to-idea link correctness, and forecastability. Automated validation can establish only structure and quote provenance; it cannot certify meaning, completeness, or correct granularity. Use frozen, independent human-reviewed references for semantic evaluation where available. Until such review exists, generated inventories are candidates and any model-as-judge review is diagnostic only.
