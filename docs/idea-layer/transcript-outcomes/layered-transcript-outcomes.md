# Layered transcript outcomes: mentions → ideas

**Status:** versioned research contract and 2019 pilot. It does not replace the active weak phrase labels or the frozen forecasting baseline.

## Why separate layers

A flat atom list asks one object to preserve transcript detail, represent a general idea, and serve as a forecast target. Adding specific people, examples, ceremonial material, or supporting claims may improve apparent coverage while making the forecast-target inventory noisier. Omitting those items makes the inventory look cleaner but less complete. The recent targeted repair showed this tension: an independent model judged completeness higher after additions, but also flagged some additions as questionable.

Keep these as linked but separately queryable objects:

1. **Transcript mention:** a concise, local representation of one substantive claim, position, action, example, or event-specific detail, with an exact quote and target-speaker turn ID. `role` and `salience` describe its function in the transcript. This is a coverage/evidence layer, not a flat list of forecast labels.
2. **Idea:** a higher-level concept or argument that may be a retrospective prediction target. It has its own kind, stance, prominence, and interpretive `forecastable` flag. Those are properties of the idea, not gates that discard its transcript evidence.
3. **Typed link:** joins mentions to ideas with `expresses`, `supports`, `illustrates`, `elaborates`, `qualifies`, or `contextualizes`. Several details can support one idea; a mention can connect to multiple ideas when warranted. Each idea needs transcript-linked evidence; every mention must belong to at least one idea.

```text
exact cited transcript span → mention(s) ──typed links──> idea target(s)
```

For example, a transcript's named project or statistic can be retained as a `supporting_detail`/`illustrative_example` mention linked to a broader policy idea, rather than being forced into a separate forecast label. A ceremonial item can remain in both layers while its idea is marked nonforecastable. The `forecastable` flag does not determine whether transcript evidence is preserved.

## Contract and code

- Contract/validation: `speechprefix/idea_layer/transcript_outcomes.py`
- `validate_layered_outcome(...)` checks schema, unique IDs, allowed relation/role values, resolved links, non-orphan mentions/ideas, and exact quote substrings against target-speaker turns. Schema v2 additionally requires every idea to have at least one `expresses` link; this relation is distinct from the mention's transcript-local role. It explicitly marks entailment and completeness as unvalidated.
- Extractor v2.3 repairs a quote only when its normalized word-token sequence has one unique exact match in the cited turn, then stores the actual transcript substring and records the repair. It drops a nonmatching extra citation only when that mention retains another exact citation; a mention with no valid citation still fails. This is provenance hygiene, not semantic verification.
- `idea_targets(...)` projects the abstract idea layer for training/evaluation; `forecastable_only=True` is an optional separate view.
- `mentions_for_idea(...)` follows idea links back to the specific supporting transcript units.
- Extractor: `research/idea_layers/outcomes/extract_layered_transcript_outcomes.py`; failed event rows can be retried and merged immutably with `research/idea_layers/outcomes/merge_layered_transcript_outcome_retries.py`.

The contract is structural and provenance-focused. A perfect schema check or exact quote match does not establish that a paraphrase is correct, that a link is semantically appropriate, or that an important idea/detail was not omitted.

A separate prototype now freezes direct-expression anchors in their own source-bound artifact before a second engine proposes optional fine-grained relations. It leaves the v2.3 joint extractor and its artifacts unchanged. The three-event pilot and subsequent 12-event 2020 quality-bar test found anchor disagreement, relation-model validation failures, and model-judged edge uncertainty; neither is a semantic quality gate. See [`two-stage-relation-annotation-prototype.md`](../relations/two-stage-relation-annotation-prototype.md) for schemas and the initial pilot, [`relation-label-quality-bar-v1.md`](../relations/relation-label-quality-bar-v1.md) for frozen criteria and the newer failed/blocked test, [`fine-relation-taxonomy-v2.md`](../relations/fine-relation-taxonomy-v2.md) for the definitions used in the failed 2021 diagnostic, and [`fine-relation-taxonomy-v3.md`](../relations/fine-relation-taxonomy-v3.md) for the stricter, not-yet-evaluated textual-warrant rules.

## Pilot

The transcript-only extractor processed the 49 2019 records that had no matches from `compact-issue-phrases-v2`. The immutable merged pilot contains 655 mentions, 377 linked idea nodes, 707 links, and 1,090 exact quote citations across all 49 events. Of the 377 idea nodes, 188 connect to multiple mentions (2–10 each); the remaining 189 have one linked mention. The extractor marked 214 ideas forecastable. Four initial responses failed structural/quote validation; three were replaced by exact-quote-valid retries from `gpt-6-luna`, and one stubborn quote case was re-extracted with `gpt-6-astra`. All rows in the final artifact pass structural and exact-substring checks. This demonstrates a workable multi-resolution representation, not that the mention lists are complete, that quotes entail every paraphrase, or that forecastability is correct.

The extraction used Pi CLI with provider `openai-codex`. Full target-speaker transcripts were sent to that configured provider; data and generated artifacts remain Git-ignored/local under `data/source-archives/`. Both model aliases are mutable. The pilot is a selected 2019 development slice, not an untouched evaluation set. The existing 49-event flat atom pilot remains unchanged; no active classifier was trained or scored on either representation.

A subsequent v2.3 extraction of the same 49-event development slice, its immutable cross-model retry chain, and the all-rows mechanical validation summary are documented in [`layered-transcript-outcomes-v2-development.md`](layered-transcript-outcomes-v2-development.md). A separate chunked extractor prototype mechanically validated one long 2019 event; its design and limits are in [`chunked-layered-outcomes-prototype.md`](chunked-layered-outcomes-prototype.md). Both are separate development artifacts and do not supersede or modify the earlier pilot.

## Evaluation should be multi-objective

Do not collapse these dimensions into one weighted “quality” score:

- **Evidence integrity:** exact quote/turn validation (deterministic substring check).
- **Mention coverage:** are salient transcript-local units represented? A model-as-judge score is diagnostic, not measured recall without an agreed reference inventory.
- **Idea quality:** support, abstraction level, atomicity, stance/type, and duplicate grouping.
- **Link quality:** does each mention actually express/support/illustrate the linked idea?
- **Forecast-target suitability:** does the idea layer represent a useful outcome for prediction? Assess separately from whether it happened in the transcript.

Report event-level tradeoffs and score thresholds. A broad mention layer and a conservative idea layer can coexist, but only an explicit idea definition and independent reference labels can establish true completeness/accuracy. With the no-human-review constraint, cross-model audits and structural checks remain diagnostics, not semantic gold. Do not iterate to a model-auditor score of 100; the evaluator can be gamed, and added details may be valid mentions but poor forecast targets.

The target-event transcript and derived outcome layers remain labels only. They must never enter that same event's pre-event context features. The seven-day model and earlier evaluation results are unchanged.

## Local reproducibility

- Validated layered artifact: `data/source-archives/2019-layered-transcript-outcomes-v1-validated.jsonl` (Git-ignored)
- Artifact SHA-256: `57e0529752b4a932fdb194f8c1acae582f4d53d80406313c115249b1c5e3e25e`
- Manifest SHA-256: `1050bb1f45e674fca7ad070ef2efc6f10fd9662f8b4bad116b5da2c876e177a4`
- Contract SHA-256: `e7494a218094574a42ef0de2f6445b2ae3815575f36348be05f5031e993c2965`
- Extraction runner SHA-256: `c4bb3d5edd071a99f9d45ce070927a7d363c4c7486963318cb69736bbd106e08`
- Retry merge runner SHA-256: `fdfaf8faee363c445f4099e391a72a242c10ee0abf1c875ae7beb303f101c57b`
- v2.3 development artifact and manifest hashes are recorded in [`layered-transcript-outcomes-v2-development.md`](layered-transcript-outcomes-v2-development.md); both files are Git-ignored/local.
