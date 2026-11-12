# Documentation

These documents describe the active retrospective context-to-ideas project. Repository contribution and agent delivery requirements are in [`../AGENTS.md`](../AGENTS.md).

- [`../README.md`](../README.md) — objective, leakage boundary, quick start, and repository map.
- [`idea-layer/README.md`](idea-layer/README.md) — canonical task definition, current data/result, and limitations.
- Layer indexes: [transcript outcomes](idea-layer/transcript-outcomes/README.md), [talking points](idea-layer/points/README.md), [topics](idea-layer/topics/README.md), [stories](idea-layer/stories/README.md), [prediction](idea-layer/prediction/README.md), [relations](idea-layer/relations/README.md), [evaluation](idea-layer/evaluation/README.md), [sources](idea-layer/sources/README.md), and [governance](idea-layer/governance/README.md).
- [`idea-layer/layered-transcript-outcomes.md`](idea-layer/transcript-outcomes/layered-transcript-outcomes.md) — transcript evidence/mention layer, idea targets, typed links, and initial pilot limitations.
- [`idea-layer/two-stage-relation-annotation-prototype.md`](idea-layer/relations/two-stage-relation-annotation-prototype.md) — separate direct-expression anchors and fine-relation candidate-edge prototype.
- [`idea-layer/relation-label-quality-bar-v1.md`](idea-layer/relations/relation-label-quality-bar-v1.md) — frozen no-human-review diagnostic thresholds and results on a new 2020 sample.
- [`idea-layer/fine-relation-taxonomy-v2.md`](idea-layer/relations/fine-relation-taxonomy-v2.md) — contrastive definitions used on the failed 2021 diagnostic sample.
- [`idea-layer/fine-relation-taxonomy-v3.md`](idea-layer/relations/fine-relation-taxonomy-v3.md) — stricter textual-warrant definitions, frozen but not yet evaluated.
- [`idea-layer/open-vocabulary-context-idea-generation.md`](idea-layer/prediction/open-vocabulary-context-idea-generation.md) — Codex headline-only open-vocabulary candidate-generation prototype and bounded diagnostic.
- [`idea-layer/layered-transcript-outcomes-v2-development.md`](idea-layer/transcript-outcomes/layered-transcript-outcomes-v2-development.md) — v2 development run counts, retry provenance, and mechanical checks.
- [`idea-layer/chunked-layered-outcomes-prototype.md`](idea-layer/transcript-outcomes/chunked-layered-outcomes-prototype.md) — checkpointed long-transcript chunking/consolidation prototype and its single-event limitations.
- [`idea-layer/idea-atom-definition-v1.md`](idea-layer/points/idea-atom-definition-v1.md) — operational idea unit, split/merge rules, and layer-specific evaluation implications.
- [`idea-layer/story-graph-architecture.md`](idea-layer/stories/story-graph-architecture.md) — evolving-story representation, temporal/provenance contract, and query-conditioned GNN design target.
- [`idea-layer/input-source-audit.md`](idea-layer/sources/input-source-audit.md) — input-source findings, coverage, and availability/rights caveats.
- [`idea-layer/2019-full-year-preliminary-study.md`](idea-layer/stories/2019-full-year-preliminary-study.md) — source coverage and preliminary 2019 calendar-year backtest diagnostics.
- [`idea-layer/2019-11-04-speech-news-circulation-case-study.md`](idea-layer/stories/2019-11-04-speech-news-circulation-case-study.md) — backwards analysis of all candidate ideas in one speech and their pre-event news/account-post time series.
- [`architecture.md`](architecture.md) — active model/data flow and code entry points.
- [`../speechprefix/stages/README.md`](../speechprefix/stages/README.md) — separate topic, idea, sentence, word, and character model-stage contracts and implementation status.
- [`evaluation.md`](evaluation.md) — chronological splits, metrics, and current development result.
- [`idea-error-analysis.md`](idea-error-analysis.md) — per-event misses, qualitative failure diagnosis, and prioritized fixes.
- [`data-audit.md`](data-audit.md) — corpus provenance and label limitations.
- [`idea-layer/evaluation-protocol.md`](idea-layer/evaluation/evaluation-protocol.md) — evaluation requirements.
- [`idea-layer/ai-review-policy.md`](idea-layer/governance/ai-review-policy.md) — constraints for assistant-reviewed semantic references.

## Keeping documentation aligned

- `idea-layer/README.md` is the single canonical task specification. Root `README.md` is the user-facing summary; `CONTEXT.md` is the short agent handoff. If the task changes, update the canonical spec first, then these summaries, the evaluation contract, and data audit in the same change.
- Document only the pre-event news/context→speech-ideas workflow as the project objective. The seven-day headline classifier is the frozen baseline; story-graph representations are separate experiments within that objective. Historical phrase-recurrence, speech continuation, live/prospective capture, and speech-plan experiments are not alternate project goals. Their source code remains for historical compatibility; CLI help must label those commands as legacy.
- Remove superseded docs; do not preserve stale instructions in an archive. Keep only docs linked from this map.
- Before completing a change, run `uv run pytest -q`, `git diff --check`, and `uv lock --check`. The documentation contract tests check local Markdown links, the canonical objective, project metadata, and legacy CLI labels. These checks catch known drift, but a reviewer must still check meaning when the objective changes.
