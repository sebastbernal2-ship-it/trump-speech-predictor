# Standalone idea-forecasting layer

## Goal

For one upcoming speech, rank **what ideas Trump is likely to address**, using only information verified as available before that speech. The most important inputs are current events and Trump’s own recent public posts; known event details are supporting context. Past speech history is a baseline or a secondary signal—not the main source of what matters now.

As the speech unfolds, the same layer may update the idea ranking using only transcript turns already observed. This is still an idea forecast. It does not predict wording or generate a speech.

## Scope boundary

This package is intentionally separate from word continuation, generic speech generation, hierarchical speech plans, and question-answer response prediction. Those remain independent prototypes. Nothing in the idea layer should import their model weights or training data.

The standalone Python boundary is `speechprefix/idea_layer/`. It now has provenance/leakage validation plus an append-only source ledger and immutable input-snapshot freezer. It contains no news/social collector or ranking model yet. Record formats are specified in [`contracts.md`](contracts.md). Legacy `context-news-*` and `context-issue-*` commands remain exploratory and are not the new system.

## Forecast lifecycle

1. **Register the event.** Record the scheduled start and a prediction cutoff. Store only event facts known before that cutoff, each linked to its source.
2. **Capture evidence.** Keep breaking-news items, Trump posts, and event metadata in an append-only, source-attributed ledger. For each item, keep publication time, first-observed time, original URL/account/post ID, exact content fingerprint, and reuse terms. `idea-source-import` validates and appends local JSONL records idempotently; it is not an automated scraper.
3. **Freeze the inputs.** `idea-snapshot-freeze` rejects items published or first observed at/after cutoff, or whose pre-cutoff availability is unverified. It saves an immutable source bundle and hash before any forecast. The default local ledger is `data/idea-layer/source-ledger.jsonl` (Git-ignored).
4. **Rank ideas.** Produce open-vocabulary human-readable ideas—not just phrases copied from headlines. Every forecast idea cites the evidence that supports it. Keep a ranking score distinct from a calibrated probability.
5. **Update during the speech.** Append new forecast snapshots as turns become observable. A change must cite already-visible turn IDs; never rewrite the initial pre-speech forecast.
6. **Score afterward.** Only the evaluator reads the completed transcript. Compare forecasts to transcript-grounded idea labels and report both hits and misses.

The contracts enforce these boundaries but cannot prove a source archive’s timestamps are truthful. Source capture procedure and evidence must be audited too.

## Data and model rules

- No transcript, transcript-derived labels, post-event notes, or future turns in pre-speech input or forecast-generation code.
- Every source has a stable ID, URL/permalink, publisher/account, original publication time, first-observed time, content hash, and reuse note. Trump posts must retain platform account and post ID.
- Same-day material is eligible only if its exact publication and observation times are strictly before cutoff. A date-only label is insufficient.
- Do not infer that a news item is relevant merely because it is recent. Rank *the likelihood Trump will address it*, using topical relevance, repeated coverage, his own posts, event-specific context, and history as separate, inspectable signals.
- Preserve original idea wording and evidence citations. Consolidation into broader ideas must be auditable; do not silently collapse Gaza, Iran, Pakistan, oil, and the Middle East into one label.
- Scores are rankings until calibrated on a separate chronological sample. Report the number of ideas forecast and how many were missed; abstention alone is not success.
- Assistant-generated transcript annotations are attributed weak references, not human ground truth. Do not report them as semantic accuracy.

## Evaluation contract

The primary task is **pre-speech, event-specific issue/idea ranking**. Each forecast is frozen at one declared cutoff. Evaluate on chronological future events, with whole events kept together. Required comparisons:

1. event metadata only;
2. recent news only;
3. Trump-posts only;
4. speaker-history only;
5. combined evidence.

Report precision@K, recall@K, nDCG@K, event coverage/abstention, and date-cluster uncertainty. Show source ablations so we can tell whether posts and current news add value. Score candidate ideas at semantic level only against cited, transcript-grounded labels; separately show lexical matching as a diagnostic. A prospective forecast score requires a new event whose source snapshot was frozen before its cutoff. The inspected 2026 holdout is not available for tuning or claims.

Live-update evaluation is a separate replay: freeze the pre-speech snapshot, reveal transcript turns in timestamp order, and score each saved update against ideas supported by turns observed by that moment. Do not mix these scores with the pre-speech result.

## Current state and next work

- Existing Google News searches are retrospective, and no prospectively frozen sample exists.
- No automated news collector or Trump-post archive is connected. Choose permitted, attributable sources and verify timestamps/reuse terms before implementation.
- The old context ranker found source-supported phrases but did not show that Trump would discuss them; the speaker-history ranker predicts recurring labels and does not serve the current-events goal.
- Next: implement source adapters for a predeclared news feed and a permitted Trump-post export. Then freeze one truly future event snapshot. Do not train a complex ranker until input provenance and an evaluation target are working.

## Repository map

- `speechprefix/idea_layer/`: isolated contracts and source ledger/snapshot freezer; source adapters, ranking, and incremental updates remain future work.
- `tests/test_idea_layer_contracts.py`: leakage, timestamps, evidence, and update-citation checks.
- `docs/idea-layer/README.md`: canonical scope and evaluation contract.
- `docs/context-theme-forecast.md`: historical experiments only.
- `speechprefix/` modules outside `idea_layer/`: existing research prototypes and baselines, not dependencies of this layer.
