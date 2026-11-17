# Transcript outcomes

This folder defines and audits post-event transcript outcomes: quote-grounded **mentions/evidence**, candidate **talking-point (`ideas`) nodes**, and typed links between them. It is a joint extraction contract, not a feature-input folder.

- [`layered-transcript-outcomes.md`](layered-transcript-outcomes.md) — contract and initial pilot.
- [`layered-transcript-outcomes-v2-development.md`](layered-transcript-outcomes-v2-development.md) — extraction/retry diagnostics.
- [`chunked-layered-outcomes-prototype.md`](chunked-layered-outcomes-prototype.md) — long-transcript strategy and limits.
- [`layered-transcript-outcome-cross-model-audit-v1.md`](layered-transcript-outcome-cross-model-audit-v1.md) — AI diagnostic, not semantic gold.
- [`transcript-processing-plan.md`](transcript-processing-plan.md) — proposed data-readiness workflow across surface and semantic targets, inventory limitations, and open decisions.
- [`mechanical-separation.md`](mechanical-separation.md) — implemented versioned word, period-split sentence, and character separation.

**Leakage rule:** all values derived from the target speech are outcome labels, never same-event prediction inputs.

The initial local whitespace cleaner and evidence-linked lexical topic/idea candidates are described in [transcript-candidates.md](transcript-candidates.md). These are unreviewed and incomplete semantic outcomes.

[Speaker selection and independent references](speaker-selection-and-references.md) describes the review-driven turn selection interface and the remaining annotation gate.
