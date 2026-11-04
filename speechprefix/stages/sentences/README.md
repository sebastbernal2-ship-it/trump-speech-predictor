# Sentences stage (reserved)

**Status:** scaffold only; no sentence model, training data, or evaluation task is implemented.

Before implementation, define a versioned input/output contract with the ideas stage, allowable context inputs, target construction, leakage rules, and sentence-level metrics. The stage should consume a frozen idea-forecast artifact; it must not silently change or rescore the idea forecast. Keep sentence realization metrics separate from idea coverage and semantic-match metrics.
