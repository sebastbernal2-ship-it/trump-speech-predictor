# Documentation

These documents describe the active retrospective context-to-ideas project. Repository contribution and agent delivery requirements are in [`../AGENTS.md`](../AGENTS.md).

- [`../README.md`](../README.md) — objective, leakage boundary, quick start, and repository map.
- [`idea-layer/README.md`](idea-layer/README.md) — canonical task definition, current data/result, and limitations.
- [`idea-layer/story-graph-architecture.md`](idea-layer/story-graph-architecture.md) — evolving-story representation, temporal/provenance contract, and query-conditioned GNN design target.
- [`idea-layer/input-source-audit.md`](idea-layer/input-source-audit.md) — input-source findings, coverage, and availability/rights caveats.
- [`idea-layer/2019-full-year-preliminary-study.md`](idea-layer/2019-full-year-preliminary-study.md) — source coverage and preliminary 2019 calendar-year backtest diagnostics.
- [`architecture.md`](architecture.md) — active model/data flow and code entry points.
- [`evaluation.md`](evaluation.md) — chronological splits, metrics, and current development result.
- [`idea-error-analysis.md`](idea-error-analysis.md) — per-event misses, qualitative failure diagnosis, and prioritized fixes.
- [`data-audit.md`](data-audit.md) — corpus provenance and label limitations.
- [`idea-layer/evaluation-protocol.md`](idea-layer/evaluation-protocol.md) — evaluation requirements.
- [`idea-layer/ai-review-policy.md`](idea-layer/ai-review-policy.md) — constraints for assistant-reviewed semantic references.

## Keeping documentation aligned

- `idea-layer/README.md` is the single canonical task specification. Root `README.md` is the user-facing summary; `CONTEXT.md` is the short agent handoff. If the task changes, update the canonical spec first, then these summaries, the evaluation contract, and data audit in the same change.
- Document only the pre-event news/context→speech-ideas workflow as the project objective. The seven-day headline classifier is the frozen baseline; story-graph representations are separate experiments within that objective. Historical phrase-recurrence, speech continuation, live/prospective capture, and speech-plan experiments are not alternate project goals. Their source code remains for historical compatibility; CLI help must label those commands as legacy.
- Remove superseded docs; do not preserve stale instructions in an archive. Keep only docs linked from this map.
- Before completing a change, run `uv run pytest -q`, `git diff --check`, and `uv lock --check`. The documentation contract tests check local Markdown links, the canonical objective, project metadata, and legacy CLI labels. These checks catch known drift, but a reviewer must still check meaning when the objective changes.
