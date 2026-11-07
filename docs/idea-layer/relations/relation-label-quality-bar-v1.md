# Provisional quality bar for transcript relation labels

**Purpose:** set a no-human-review operational screen before evaluating a fresh sample. This is a consistency/grounding bar for research graph candidates, **not** a semantic-accuracy standard. No model-only threshold can certify that labels are true. Passing would not authorize same-event prediction features or establish forecastability.

## Frozen gates

A label set may be promoted from an unreviewed prototype to **candidate graph research** only if every gate below passes on a frozen sample of at least 12 events, selected without examining transcript semantics or annotation outputs. These thresholds are provisional engineering tolerances, not empirically calibrated error rates.

### 1. Structural and provenance gate — 100%

- Every outcome, anchor, relation, and audit row validates against the frozen source event and transcript hashes.
- Every idea is either linked to one or more direct-expression anchors or explicitly recorded as unanchored; no forced anchor is required.
- Every candidate fine edge resolves to source-quoted mention evidence and an anchored idea, uses one allowed non-`expresses` type, and preserves source/model provenance.
- No silent repair, dropped invalid row, or model-error row is counted as a pass.

### 2. Anchor consistency screen

Run the same frozen prompt/configuration against two different model aliases on the same frozen outcome inventory. Compare exact `(mention_id, idea_id)` pair sets per event.

- Mean per-event Jaccard similarity must be at least **0.90**; every event must be at least **0.80**.
- In a separate omission/directness review, at least **95%** of supplied anchors must be judged direct, and model-suggested missing pairs must be no more than **5%** of the primary inventory size.
- Report how many suggested omissions recur in the second inventory, but do not automatically accept them. This is a cross-model consistency diagnostic, not recall.

### 3. Fine-relation screen

- At least **100** candidate fine edges must be reviewed in aggregate.
- The reviewer must be from a **different provider** than both extractors; model-alias changes within one provider do not satisfy this gate.
- At least **90%** of reviewed edges must be judged plausible, at most **10%** uncertain, and at most **5%** mismatched.
- A relation type with fewer than **20** reviewed examples remains exploratory and must not be treated as validated, regardless of aggregate scores.

### 4. Scope boundary

Even if all automated gates pass, the result is only eligible for further candidate-graph experiments. Semantic accuracy still cannot be certified without independent reference labels or human review. Transcript-derived anchors/edges remain excluded from same-event prediction inputs; chronological news/context evaluation and point-in-time source availability are separate gates.

If any gate fails or is blocked, keep the annotations as versioned candidates and do not promote them to trusted graph facts or forecasting features. Report the failure rather than tuning prompts against the sample.

## Previous frozen test: 2020 taxonomy-v1 sample (failed)

The earlier 2020 sample is frozen in `data/source-archives/2020-two-stage-quality-sample-v1.freeze.json` (SHA-256 `fde0e90a610c5d436edba0b66e35377c33ce0d0497a50f21157d4338cba2b0d7`). It contains 12 events selected by date, length, and event-ID hash only. Its failures and original v1 output hashes remain unchanged and were not rerun or tuned:

- Outcomes validated on 12/12 events (452 mentions, 330 ideas). Anchor passes produced 390 and 399 pairs; mean exact-pair Jaccard was 0.916, but the range was 0.719–1.000, failing the event floor. The same-provider anchor review returned 269 direct and 1 uncertain on 270 anchors; its 14/270 omission suggestions (5.2%) exceeded the threshold, with only 5/14 in pass B.
- Fine edges succeeded on 10/12 events (96 edges, below the 100-edge minimum). The same-provider reviewer marked 70 plausible (72.9%), 12 uncertain (12.5%), 14 mismatched (14.6%), missing all provisional rate gates. Two extraction failures were recorded, but the original v1 runner omitted failed-event rows.
- Gate decision: fail. Model/provider aliases are all OpenAI Codex; semantic accuracy and cross-provider review are unavailable.

The frozen output hashes are retained: outcomes `55b7f83c8f10fac431e6b84fded2af54ab13eb95c780605160d30ee4a1494263`; anchors A `ff0fa86f58c7791a64d19b20fdcac3cda3da3a8b3f9a89cb8a7d40ebb725279a`; anchors B `080380640dc632b84dcb67e4cb922f0a9fa3430e94017a21ab65f30df2049908`; relations `88126eb3c086fcf437be94d53bee4b7333524bb47ff4af853e30d71be9e4b471`; review `a26e74d63eff47566adf92d7e0d392fed1734a74e02a7aa0cbb49bc6f927c3ee`.

## New frozen test: 2021 taxonomy-v2 sample

The selected sample is `2021-relation-taxonomy-v2-quality-sample-v1`, frozen in local ignored file `data/source-archives/2021-relation-taxonomy-v2-quality-sample-v1.freeze.json` (SHA-256 `b41b1647b61c2f211b5a2359cd561fe44897cd8c701eae169729bbb1e97406e0`). The source event corpus (`data/rollcall-by-type/trump-speech-candidates.jsonl`) has SHA-256 `c7e337830eddf0c1213759f87045369b862d2a74a5c1d89515912f84819a1fdf`. The sample is a census of all 2021 events with at least 1,000 target-speaker transcript words: **17 events**. Eligibility uses only date and word count; there are no outcome, relation, or audit scores in the rule.

The event IDs are `dt45-21-8d7f8f02`, `dt45-21-3ecc538e`, `dt45-21-2a8939f7`, `dt45-21-e8b78ea8`, `dt45-21-3528250d`, `dt45-21-1dc4abbd`, `dt45-21-4a112a59`, `dt45-21-5f8b09c3`, `dt45-21-f781108e`, `dt45-21-e782ee96`, `dt45-21-f3e018e6`, `dt45-21-dd4e36a9`, `dt45-21-425891d5`, `dt45-21-f5e02e74`, `dt45-21-7ad17e93`, `dt45-21-218c0f06`, and `dt45-21-7d51f37a`.

The v2 prompt was applied without revising the taxonomy based on this sample. Outcome extraction used `openai-codex / gpt-6.1-terra`; anchor inventories used `gpt-6.1-sol` and `gpt-6-luna`; fine-edge extraction used `gpt-6-luna`; the diagnostic reviewer used `gpt-6-astra`. All are aliases within the **same provider**. Full target-speaker transcripts and transcript-derived outcomes were sent to that provider. Thus the independent-provider gate is blocked; model verdicts below are diagnostics, not semantic truth.

### Results

- **Outcome extraction:** 17/17 rows structurally validated; 851 mentions and 629 ideas. These are model-created inventories, not verified completeness.
- **Anchor repeatability:** pass A produced 686 pairs and pass B 692. Mean per-event exact-pair Jaccard was **0.943**; minimum **0.836**; all 17 exceed the provisional 0.80 per-event floor. This passes the consistency thresholds only. A reviewer examined **551** pass-A anchors on 14 events (not all 17): 547 `direct`, 3 `uncertain`, 1 `mismatch` (99.3% direct among reviewed). It suggested 13 possible missing pairs; **0/13** appeared in pass B. The suggestion rate is 13/686 = **1.9%** of the primary pass-A inventory, below the provisional 5% threshold. Both the directness and suggestion-rate screens therefore pass as same-provider diagnostics only; the zero overlap and bounded scan do not establish recall.
- **Fine-edge extraction:** all 17 event rows were structurally successful after one recorded repair; 298 candidate edges were produced. Type counts: 228 `supports`, 38 `illustrates`, 22 `elaborates`, 10 `contextualizes`, and 0 `qualifies`.
- **Fine-edge diagnostics:** the reviewer returned complete verdicts for 257 edges across 14 events; 165 plausible (**64.2%**), 51 uncertain (**19.8%**), and 41 mismatched (**16.0%**). The provisional rate gates fail. By type (plausible / uncertain / mismatch): `supports` 135/31/34 (n=200), `illustrates` 16/8/3 (n=27), `elaborates` 12/5/3 (n=20), `contextualizes` 2/7/1 (n=10), `qualifies` 0/0/0. `contextualizes` and `qualifies` remain exploratory for insufficient reviewed examples.

Per-event anchor Jaccard and edge counts/verdicts are retained in the local audit JSONL and its manifest. Pairwise Jaccard uses exact `(mention_id, idea_id)` pairs and is macro-averaged across all 17 events. Three event rows have `model_error` status after retries, with zero returned anchor/edge verdicts; they are not silently interpreted as positive judgments. Review rates use only returned, structurally complete verdicts. The 13 omission suggestions do not overlap with pass B, which is a disagreement diagnostic, not evidence that either inventory is complete.

**Gate decision: failed; do not promote.** The anchor-set consistency gate passes, and sampled anchor-directness judgments clear the provisional threshold, but fine-edge plausibility/uncertainty/mismatch thresholds fail substantially. Three events have no reviewer verdicts; reviewer-provider independence is unavailable. The sample is only 17 events and cannot establish semantic accuracy or label recall. Keep edges as research candidates, not trusted graph facts or forecasting features.

After recording this result, stricter textual-warrant rules were frozen as [`fine-relation-taxonomy-v3.md`](fine-relation-taxonomy-v3.md). V3 has not been run on these data; it cannot retroactively improve the v2 result and must be tested on a newly frozen sample.

### Local-only artifact hashes

| Artifact | SHA-256 |
|---|---|
| `2021-relation-taxonomy-v2-quality-sample-v1.freeze.json` | `b41b1647b61c2f211b5a2359cd561fe44897cd8c701eae169729bbb1e97406e0` |
| `2021-layered-outcomes-taxonomy-v2-quality-sample-v1.jsonl` | `0835240992925c698a1d15ee6954fe9e057869f076cd303a971154c96a0eb742` |
| `2021-layered-expression-anchors-taxonomy-v2-pass-a-v1.jsonl` | `782f50e18f59d53e511f0fdcc342eb777af6694e2c2e1d31a4f7f03bb646ad4a` |
| `2021-layered-expression-anchors-taxonomy-v2-pass-b-v1.jsonl` | `772001309011bc94d870394bd9cbbfa3900288e88227a67c4dec76a9c59db2b5` |
| `2021-layered-fine-relations-taxonomy-v2-quality-sample-v1.jsonl` | `27c71836c3d110e8c924f64cb30297c677b8599f447f164a4899215dad26c70a` |
| `2021-layered-fine-relations-taxonomy-v2-review-v1.jsonl` | `e083bfdd088492f58c763049c5de7cfb01ec5279bb99d8f3fa3afa919ddc9f46` |

## Provider limitation and next gate

At the earlier freeze, DeepSeek returned HTTP 402 `Insufficient Balance` and Anthropic was not configured. The user requested that this continuation use Codex, so all present calls intentionally used `openai-codex`; this cannot satisfy the distinct-provider review gate. A different Codex alias is not an independent provider. Do not silently waive the gate: either configure/fund an authorized second provider for a future sample, or explicitly maintain the blocked status. Even cross-provider model review is not semantic gold; without independent reference labels or human review, accuracy cannot be certified.
