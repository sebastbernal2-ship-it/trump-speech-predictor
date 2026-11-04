# Documentation

These documents describe the active retrospective context-to-ideas project:

- [`../README.md`](../README.md) — objective, leakage boundary, quick start, and repository map.
- [`idea-layer/README.md`](idea-layer/README.md) — canonical task definition, current data/result, and limitations.
- [`architecture.md`](architecture.md) — active model/data flow and code entry points.
- [`evaluation.md`](evaluation.md) — chronological splits, metrics, and the current development result.
- [`data-audit.md`](data-audit.md) — corpus provenance and label limitations.
- [`idea-layer/evaluation-protocol.md`](idea-layer/evaluation-protocol.md) — evaluation requirements.
- [`idea-layer/ai-review-policy.md`](idea-layer/ai-review-policy.md) — constraints for assistant-reviewed semantic references.

## Keeping documentation aligned

- `idea-layer/README.md` is the single canonical task specification. Root `README.md` is the user-facing summary; `CONTEXT.md` is the short agent handoff. If the task changes, update the canonical spec first, then these summaries, the evaluation contract, and data audit in the same change.
- Document only the active context-headlines→speech-ideas workflow as the project objective. Historical phrase-recurrence, speech continuation, live/prospective capture, and speech-plan experiments are not alternate project goals. Their source code remains for historical compatibility; CLI help must label those commands as legacy.
- Remove superseded docs; do not preserve stale instructions in an archive. Keep only docs linked from this map.
- Before completing a change, run `uv run pytest -q`, `git diff --check`, and `uv lock --check`. The documentation contract tests check local Markdown links, the canonical objective, project metadata, and legacy CLI labels. These checks catch known drift, but a reviewer must still check meaning when the objective changes.
