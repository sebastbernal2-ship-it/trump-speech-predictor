# Cross-model audit of transcript idea atoms

**Status:** AI-only development diagnostic. This is not human review, semantic gold, or an accuracy estimate.

## Result

A second model audited all 269 candidate atoms from the 49-event pilot against each event's complete available target-speaker transcript. The extractor's event IDs and transcript turn IDs were blinded during the audit; no event date/title, news, forecast, or old weak labels were supplied. The auditor was not shown the extractor's confidence or prominence fields. It used provider `openai-codex`, model alias `gpt-6.1-sol`, different from the extractor alias `gpt-6-luna` but still within the same provider family. The full target-speaker transcripts and candidate atoms were sent to the configured inference provider for this audit; generated audit outputs remain local and Git-ignored.

**Composite cross-model audit score: 95.34/100** (49-event macro mean; median 95.00; range 88.54–100). All 49 audits returned valid results. This is version 2 of the audit; a preliminary version that exposed confidence/prominence was retained locally but is not used for this score. The rubric weights were fixed in the audit code:

| Dimension | Weight | Event-macro mean | Normalized score |
|---|---:|---:|---:|
| Exact quote integrity (mechanical substring check) | 5% | 440/440 exact | 100.0/100 |
| Evidence support | 35% | 3.84/4 | 96.0/100 |
| Atom granularity | 20% | 3.90/4 | 97.5/100 |
| Summary and stance fidelity | 20% | 3.95/4 | 98.7/100 |
| Type/forecastability fit | 10% | 3.81/4 | 95.3/100 |
| Event-level completeness | 10% | 3.18/4 | 79.6/100 |

The composite mostly indicates that the auditor found the *atoms that were emitted* grounded and reasonably formed. It does **not** mean that 95.34% (or 269/269) are correct. Completeness receives only 10% of the composite, and the audit flagged a possible major omission in 29/49 event inventories. The score should not be used without that qualification.

The auditor attached at least one issue flag to 23/269 atoms: 12 kind mismatches, 7 cases of conflated ideas, 3 stance mismatches, 4 forecastability mismatches, 1 understatement, and 1 vague atom (flags can overlap). Five atoms received evidence-support scores at or below 2/4 (one scored 1, four scored 2); none scored 0. These are model judgments, not independently verified error counts. The most important observed weakness is **incomplete inventories**: the auditor flagged a possible major omission in 29/49 events. Occasional category/stance mistakes and merging of distinct points are secondary issues.

## Rubric and limitations

Each atom received 0–4 ratings for evidence support, atomicity, summary/stance fidelity, and type/forecastability fidelity. Each event received a 0–4 completeness rating. The composite gives 5/35/20/20/10/10 percent respectively to quote integrity, evidence support, atomicity, summary/stance, type/forecastability, and completeness. Events are equally weighted for the aggregate. The report also retains per-event scores, atom-level reasons/flags, and candidate omissions for diagnosis.

The judge is another AI model—not human adjudication—and shares the inference provider with the extractor. Both model aliases are mutable and are not immutable checkpoint identifiers. Different aliases reduce direct self-grading but do not remove shared biases, correlated errors, generous ratings, or hallucinated omissions. Transcript identity may also be inferable from content despite metadata blinding. Exact quote matches establish location, not semantic entailment. The completeness judgments are particularly subjective. No semantic gold exists, and no claim of accuracy should be inferred from this score.

## Targeted completeness-repair experiment

The first audit's 29 inventories rated below 4/4 completeness were sent through a targeted supplement pass. The extractor (`gpt-6-luna`) received each complete transcript, the existing atoms, and the auditor's possible omissions as **unverified leads**; these were sent to the configured inference provider. It was instructed to reject unsupported/detail-only leads, preserve the old atoms, cite exact turns, and return no additions if none were warranted. It added 37 atoms across 19 events; 10 of the 29 cases returned no additions. The merged artifact is a new immutable version.

To avoid scoring the repair with the model that supplied the leads, a third alias (`gpt-6.1-terra`) audited the original and augmented inventories with the same blind rubric and no access to the suggestions. This is a paired model-judge comparison, not a gold-standard experiment:

| Holdout auditor diagnostic | Original (269 atoms) | After supplement (306 atoms) |
|---|---:|---:|
| Event-macro composite | 95.48/100 | 96.09/100 |
| Event-macro completeness | 3.31/4 (82.7/100) | 3.63/4 (90.8/100) |
| Events rated complete (4/4) | 24/49 | 34/49 |
| Events with possible major omissions | 25/49 | 15/49 |
| Exact quote matches | 440/440 | 493/493 |

Completeness improved in 14 events and declined in 2; 33 were unchanged. Of the 37 added atoms, the holdout rated 28 evidence-support 4/4, 7 as 3/4, and 2 as 2/4; it attached an issue flag to 7 additions, including one it called unsupported. Atom-level evidence/summary/type means dipped slightly while completeness rose. This is the precision–coverage tradeoff the composite can hide. The earlier auditor's baseline completeness was 3.14/4 versus the holdout's 3.31/4, so even model raters differ before any repair.

**Practical assessment:** the targeted pass materially improved *model-judged inventory coverage* (82.7 → 90.8/100 under the same holdout alias) while introducing some questionable additions. It is promising for generating transcript-grounded candidates, but 15/49 inventories remain below the holdout's completeness threshold. A judge rating of 4/4 means that judge did not notice a major omission; it is not proof that none exists. Do not iterate until a judge returns 100 or use this selected subset as uniform training/evaluation labels. First freeze the definition of an idea atom—especially whether named people, supporting anecdotes, and ceremony details count separately—then validate a consistent extraction protocol across the chronological corpus. Without human/external gold, a true semantic accuracy or recall score remains unavailable.

## Reproducibility

The audit runner is [`../../research/audit_transcript_idea_atoms.py`](../../research/audit_transcript_idea_atoms.py); targeted augmentation and immutable merge are in [`../../research/augment_transcript_idea_atoms.py`](../../research/augment_transcript_idea_atoms.py). These use blind transcript-only Pi CLI calls and exact-quote/provenance checks. The baseline v2 audit, paired holdout audits, supplement, and merged candidate inventory remain local/Git-ignored under `data/source-archives/`: `2019-semantic-idea-atom-quality-audit-v3-terra-baseline.jsonl`, `2019-idea-atoms-targeted-supplement-v1.jsonl`, `2019-weak-empty-semantic-idea-atoms-v2-augmented.jsonl`, and `2019-semantic-idea-atom-quality-audit-v3-terra-augmented.jsonl`.

- Extractor artifact SHA-256: `024d6033f6d9330b14752a2956427c33ab67fe040cb8af5286f34fcb6723e734`
- Input transcript corpus SHA-256: `c7e337830eddf0c1213759f87045369b862d2a74a5c1d89515912f84819a1fdf`
- Audit prompt-template SHA-256: `563ef78fe6f647c8533b3ec3058e0a970f171aa3da846b7eddac2035e04017ad`
- Audit runner SHA-256: `e49bc61ca90ad421f8e4dcfe42c3f3ee6bc52746c0105241646fb0bf3b919b0c`
- Per-event audit artifact SHA-256: `0eb814c56236e22d9c58b9c31a99e26fe6a793c9cdef9abfc449212803431d5e`
- Audit manifest SHA-256: `e41ab26f6d62be97d7d7588999a135c99b558103c8cc7d15372eebc5bbf811b4`

Paired holdout artifacts (same `gpt-6.1-terra` alias, before/after):

- Original inventory audit SHA-256: `ca87de4c4c27f20d059ede9a883fbed78d87de0bc9458a78739aa01fc4c9781e`
- Targeted supplement SHA-256: `cfbe952226acc1d073c5bbc2f35e10d3ab3bea33b94d8560b8ce5758e34bab54`
- Augmented inventory SHA-256: `bb7a617c728eb093798c9c0eb2d17f262073fd8be94a55693e26f02092893431`
- Augmented holdout audit SHA-256: `bd5b5f94879460703b4417404419e26c7b5d61fcda60624a9c4e5a4266b01995`
- Augmentation runner SHA-256: `3639c2b96fbc0f0139d228fbba000cd54b67c0e06f49828079cd6ae7c021fcc9`
