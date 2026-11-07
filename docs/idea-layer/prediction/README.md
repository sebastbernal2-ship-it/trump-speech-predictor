# Prediction tasks and input features

This folder concerns forecasts from strictly pre-event information, not target-speech extraction.

- [`multilevel-prediction-architecture.md`](multilevel-prediction-architecture.md) — separate topic, point, story, and optional expression tasks; feature/outcome boundary and chronological evaluation.
- [`open-vocabulary-context-idea-generation.md`](open-vocabulary-context-idea-generation.md) — headline-only candidate-generation diagnostic.

Each semantic level may be a prediction target. Its input features must be built from sources available before the target cutoff; transcript-derived target labels never enter same-event features.