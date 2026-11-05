# Active project context

## Objective

Build and evaluate a **retrospective supervised model** that predicts ideas in a Trump speech from pre-event news and context. The current headline model's seven-day window is a frozen benchmark, not an assumed optimal horizon; a separate story-graph research track will represent duplicates, updates, persistent stories, and time-bounded entity connections from broader pre-cutoff evidence. Use historical input→output pairs to train and tune. No upcoming speech or live deployment is required.

## Non-negotiable feature/label boundary

For event *i*:

- **Baseline features:** event *i*'s pre-event headlines from the preceding seven days. A separate story-graph experiment may retain older evidence and evolving-story links, but only as-of the target cutoff and with temporal relevance evaluated chronologically.
- **Label/outcome:** ideas expressed in event *i*'s target-speaker transcript.
- **Forbidden same-event leakage:** the target transcript, transcript-derived labels, post-event material, or future context must never be features for predicting event *i*.
- Earlier examples' context and transcript-derived labels are supervised training pairs. Earlier speech text or historical idea recurrence is not an inference feature for the target event.

The active CLI is `context-idea-classifier-evaluate`; the baseline model is in `speechprefix/stages/ideas/model.py`. The experimental graph contract is in `speechprefix/idea_layer/story_graph.py`; it does not yet cluster or score news. See `docs/idea-layer/README.md` for the objective, baselines, data boundaries, current result, and limitations. `docs/README.md` is the documentation map.

## Evaluation and claims

Use chronological splits by whole event date. Select models using training/validation data and report loss, ranking/coverage, and per-event misses on later dates. The current snapshots were retrieved retrospectively; only 38 unique context/speech pairs are available (32 with extractable labels), and the target phrase extractor is incomplete. Metrics are small-corpus development diagnostics, not semantic accuracy or proof of point-in-time availability. Report failures plainly.

Transcript idea references are AI-reviewed only when manually created; they are interpretive references, not objective semantic gold. See `docs/idea-layer/ai-review-policy.md`.
