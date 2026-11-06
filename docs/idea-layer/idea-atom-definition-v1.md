# Operational idea definition v1

**Scope:** assistant-authored retrospective transcript outcomes. This definition is a protocol for consistent weak extraction, not semantic ground truth.

## Unit of analysis

An **idea** is one distinct, independently meaningful unit of thought the speaker substantively addresses. It may be:

- a topic that receives meaningful development;
- a factual or political claim;
- an argument or explanation;
- a policy or political position;
- a proposal, commitment, appeal, or accusation;
- an event-specific subject that the speaker treats as an independent focus, including ceremony or recognition.

An idea should be summarizable in one neutral sentence based on transcript evidence, without adding motives or context from outside the transcript. Keep separate claims/positions separate when they could independently be accepted, rejected, or forecast. Do not split one argument into every supporting fact, name, statistic, anecdote, example, or rhetorical repetition. A supporting detail becomes its own idea only when the speaker substantively develops it as a distinct topic. Repetition increases prominence, not idea count.

**Forecastability is a separate idea property.** Preserve an idea because it was addressed even if it is event-specific, ceremonial, incidental, or unlikely to be forecast from pre-event context; mark such a candidate `forecastable=false` when appropriate. Do not use forecastability to decide whether the transcript mention exists. The flag is a weak model judgment, not calibrated probability.

## Linked representation

1. A **mention** records one concise transcript-local content unit with an exact contiguous quote and target-speaker turn ID. Its role (`core_expression`, `supporting_detail`, `illustrative_example`, `event_specific_detail`, `ceremonial_or_personal`, `other`) and salience are separate from idea identity.
2. An **idea** records the higher-level summary, concept, kind, stance, prominence, and forecastability.
3. A typed **link** connects a mention to an idea. Every idea needs at least one mention connected by `expresses`, meaning the mention directly states the idea; supporting mentions may link by `supports`, `illustrates`, `elaborates`, `qualifies`, or `contextualizes`. Mention role and link relation answer different questions: role classifies the transcript-local unit, while `expresses` states that it directly anchors this specific idea. A supporting detail may express a distinct idea when its content itself is the idea, but examples and evidence should not be mislabeled as direct expression. Every mention also links to at least one idea. Several mentions may support one idea, and one mention may link to multiple ideas only when it actually warrants that.

This anchor rule constrains abstraction: a model cannot create a forecast idea solely from a name, statistic, anecdote, or other supporting detail without a transcript mention that directly expresses the idea. The structural validator checks the relation/role anchor, but it cannot determine whether the anchor semantically entails the idea.

## Edge cases

- **Repeated argument:** one idea, multiple mentions or multiple quotes; increased prominence.
- **Claim plus distinct proposal:** two ideas if the proposal has its own independently meaningful content; link the relevant mentions separately.
- **Example/statistic:** mention linked as `illustrates` or `supports`; do not create a second idea unless it receives substantive independent focus.
- **Named person/event:** mention-only when used as evidence or a passing reference; an event-specific idea if the speaker treats the subject itself as a focus.
- **Ceremonial or personal content:** preserve as mentions and, when independently meaningful, a nonforecastable idea; do not discard it solely for low forecastability.
- **Ambiguous stance or abstraction:** choose a narrower transcript-faithful summary and use `unclear` rather than infer.
- **No substantive outcome or unusable transcript:** keep these as distinct statuses in extraction/audit; never turn an empty inventory into a claim that the speech had no ideas.

## Evaluation implications

Keep separate diagnostics for: exact quote integrity; mention coverage; idea grounding/atomicity; link correctness; and forecast-target suitability. The validator establishes only structure, link resolution, and exact substring provenance. No rule can establish semantic entailment or complete recall without a trusted reference inventory. The 2019 pilot is development data; no score from a model auditor is objective accuracy.
