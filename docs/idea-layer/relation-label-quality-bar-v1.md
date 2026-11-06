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

## First test sample, frozen before extraction

The selected sample is `2020-two-stage-quality-sample-v1`, frozen in local ignored file `data/source-archives/2020-two-stage-quality-sample-v1.freeze.json` (SHA-256 `fde0e90a610c5d436edba0b66e35377c33ce0d0497a50f21157d4338cba2b0d7`). The source event corpus (`data/rollcall-by-type/trump-speech-candidates.jsonl`) has SHA-256 `c7e337830eddf0c1213759f87045369b862d2a74a5c1d89515912f84819a1fdf`; its hash and each selected event's date/word count are also recorded in the freeze file. Selection is deterministic: among 2020 events with at least 1,000 target-speaker words, choose the smallest SHA-256(event ID) in each calendar-quarter × length-band stratum (`short` 1,000–2,499 words; `medium` 2,500–4,999; `long` 5,000+). This yields 12 events, one in each of the 12 strata; it uses only date, word count, and event ID. Three selected transcripts exceed 10,000 words.

The IDs are: `dt45-20-277cdb42`, `dt45-20-7519da74`, `dt45-20-c0373863`, `dt45-20-024af036`, `dt45-20-d4e2e9ce`, `dt45-20-92f7cee9`, `dt45-20-6c997ef5`, `dt45-20-e65873ce`, `dt45-20-d8f2a1e1`, `dt45-20-f760143f`, `dt45-20-8ee963f5`, and `dt45-20-aa326108`.

No outcome, anchor, edge, or audit extraction has been run on this sample at the time of freezing. The existing 2019 development sample is not reused for this test. All sample transcripts and generated artifacts remain local and Git-ignored.

## Frozen-sample result: gate failed

The 12-event sample was processed without changing the frozen prompts. Outcome extraction used `openai-codex / gpt-6.1-terra`; anchor passes used `gpt-6.1-sol` and `gpt-6-luna`; fine edges used `gpt-6-luna`; the diagnostic reviewer was `gpt-6-astra`. All are aliases under the same provider. Full target-speaker transcripts and transcript-derived outcomes were sent to that provider. These are diagnostics on model-generated labels, not semantic accuracy:

- **Outcome/anchor input:** 12/12 outcome rows were structurally validated (452 mentions, 330 ideas). Anchor pass A (`gpt-6.1-sol`) emitted 390 pairs; pass B (`gpt-6-luna`) emitted 399. Both accounted for all ideas without explicit abstentions.
- **Anchor repeatability:** exact-pair per-event Jaccard averaged **0.916**, but ranged from **0.719 to 1.000**. This passes the macro threshold but fails the per-event minimum 0.80 on `dt45-20-8ee963f5`. The two sets had 33 symmetric-difference pairs overall.
- **Anchor review:** the same-provider auditor reviewed only the 10 events with valid relation output: 269/270 supplied anchors were judged direct and one uncertain. It suggested 14 additional pairs (14/270 = **5.2%**, above the 5% limit); only 5/14 appeared in pass B. This does not estimate omission recall, and the two relation-error events received no anchor review.
- **Fine-edge extraction:** 10/12 events produced structurally valid candidate edges. Two failed after three format/validation attempts: `dt45-20-6c997ef5` repeated direct-expression anchor pairs; `dt45-20-f760143f` returned mention IDs absent from the frozen inventory. The run manifest recorded both errors, but the then-current runner omitted failed events from the relation JSONL. The 10 valid rows contained **96** edges, below the 100-edge minimum. A subsequent engine-v2 change now emits a source-bound `model_error` row for every exhausted event, records repair attempts/errors on successful rows, and distinguishes failures from a successful empty edge set; the audit can still review anchors on failed events while marking edge review as not run. The frozen 2020 artifacts above remain the original v1 run and were not overwritten or rerun.
- **Fine-edge review:** on those 96 edges, the same-provider auditor marked 70 plausible (**72.9%**), 12 uncertain (**12.5%**), and 14 mismatched (**14.6%**). All three proportions miss the specified limits (90% / 10% / 5%). Per-type counts were: `supports` 39/7/5 (plausible/uncertain/mismatch, n=51), `illustrates` 15/1/4 (n=20), `elaborates` 8/2/3 (n=13), and `contextualizes` 8/2/2 (n=12); no `qualifies` edges were proposed. Types with fewer than 20 examples remain exploratory.

**Gate decision: not good enough for graph promotion.** Structural outcome/anchor checks pass, but anchor agreement, relation extraction coverage, relation ratings, edge count, and provider independence fail or are blocked. Keep these as candidate annotations; do not use them as trusted graph facts or forecasting features. Do not tune these frozen results into a pass.

The local ignored artifacts are:

| Artifact | SHA-256 |
|---|---|
| `2020-two-stage-quality-sample-v1.freeze.json` | `fde0e90a610c5d436edba0b66e35377c33ce0d0497a50f21157d4338cba2b0d7` |
| `2020-layered-outcomes-frozen-quality-sample-v1.jsonl` | `55b7f83c8f10fac431e6b84fded2af54ab13eb95c780605160d30ee4a1494263` |
| `2020-layered-expression-anchors-pass-a-v1.jsonl` | `ff0fa86f58c7791a64d19b20fdcac3cda3da3a8b3f9a89cb8a7d40ebb725279a` |
| `2020-layered-expression-anchors-pass-b-v1.jsonl` | `080380640dc632b84dcb67e4cb922f0a9fa3430e94017a21ab65f30df2049908` |
| `2020-layered-fine-relation-edge-candidates-frozen-quality-sample-v1.jsonl` | `88126eb3c086fcf437be94d53bee4b7333524bb47ff4af853e30d71be9e4b471` |
| `2020-layered-relation-audit-frozen-quality-sample-v1.jsonl` | `a26e74d63eff47566adf92d7e0d392fed1734a74e02a7aa0cbb49bc6f927c3ee` |

Pairwise Jaccard is calculated over exact `(mention_id, idea_id)` sets per event and arithmetically averaged. Review fractions use only returned, structurally complete verdicts; extraction failures and unreviewed events remain failures, not negative or zero-edge examples. The fixes are a new engine/prompt version, not a reclassification of this frozen result.

## Provider limitation

At freeze time, `openai-codex` authentication was ready. A minimal DeepSeek request returned HTTP 402 `Insufficient Balance`; Anthropic was not configured. Therefore the required cross-provider relation review is unavailable. The `gpt-6-astra` review above is same-provider and diagnostic only; it cannot pass Gate 3. The overall result remains **blocked/failed**, regardless of same-provider ratings.
