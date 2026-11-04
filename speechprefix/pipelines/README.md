# Pipeline orchestration (scaffold)

**Status:** no end-to-end pipeline is implemented.

Pipeline code may pass versioned artifacts between stages and report end-to-end results. It must not contain stage-specific model logic or merge the stages' evaluation metrics. Keep each stage independently runnable and testable.
