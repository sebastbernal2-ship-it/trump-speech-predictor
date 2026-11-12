# Sentences stage (reserved)

**Output unit:** one sentence-level realization/selection. This is downstream of idea-level planning and upstream of word/token prediction.

**Status:** scaffold only; no sentence model, training data, or evaluation task is implemented.

Before implementation, define a versioned input/output contract with the ideas stage, allowable context inputs, target construction, leakage rules, and sentence-level metrics. At inference, consume a frozen predicted-ideas artifact; do not silently substitute gold target ideas, change the idea forecast, or rescore it. Keep teacher-forced sentence results separate from end-to-end results and report sentence metrics separately from idea coverage and semantic matching. Downstream word and character stages must consume this stage's serialized, versioned sentence artifact.
