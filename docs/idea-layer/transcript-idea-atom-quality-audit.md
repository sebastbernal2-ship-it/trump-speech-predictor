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

The composite mostly indicates that the auditor found the *atoms that were emitted* grounded and reasonably formed. It does **not** mean that 95.66% (or 269/269) are correct. Completeness receives only 10% of the composite, and the audit flagged a possible major omission in 29/49 event inventories. The score should not be used without that qualification.

The auditor attached at least one issue flag to 23/269 atoms: 12 kind mismatches, 7 cases of conflated ideas, 3 stance mismatches, 4 forecastability mismatches, 1 understatement, and 1 vague atom (flags can overlap). Five atoms received evidence-support scores at or below 2/4 (one scored 1, four scored 2); none scored 0. These are model judgments, not independently verified error counts. The most important observed weakness is **incomplete inventories**: the auditor flagged a possible major omission in 29/49 events. Occasional category/stance mistakes and merging of distinct points are secondary issues.

## Rubric and limitations

Each atom received 0–4 ratings for evidence support, atomicity, summary/stance fidelity, and type/forecastability fidelity. Each event received a 0–4 completeness rating. The composite gives 5/35/20/20/10/10 percent respectively to quote integrity, evidence support, atomicity, summary/stance, type/forecastability, and completeness. Events are equally weighted for the aggregate. The report also retains per-event scores, atom-level reasons/flags, and candidate omissions for diagnosis.

The judge is another AI model—not human adjudication—and shares the inference provider with the extractor. Both model aliases are mutable and are not immutable checkpoint identifiers. Different aliases reduce direct self-grading but do not remove shared biases, correlated errors, generous ratings, or hallucinated omissions. Transcript identity may also be inferable from content despite metadata blinding. Exact quote matches establish location, not semantic entailment. The completeness judgments are particularly subjective. No semantic gold exists, and no claim of accuracy should be inferred from this score.

**Practical assessment:** promising for generating transcript-grounded candidate atoms; not yet reliable as a complete, uniform outcome inventory and not ready to replace the frozen lexical labels in forecasting. Before relying on it, improve/check completeness and classification across the full chronological training/evaluation corpus, then evaluate forecasts using consistently extracted outcomes. Do not train against only this selected 49-event subset.

## Reproducibility

The audit runner is [`../../research/audit_transcript_idea_atoms.py`](../../research/audit_transcript_idea_atoms.py). It makes one blind transcript-only audit request per event through the Pi CLI and validates every returned atom ID and rating. The reported per-event artifact is `data/source-archives/2019-semantic-idea-atom-quality-audit-v2.jsonl`; its sibling manifest and both files remain local/Git-ignored.

- Extractor artifact SHA-256: `024d6033f6d9330b14752a2956427c33ab67fe040cb8af5286f34fcb6723e734`
- Input transcript corpus SHA-256: `c7e337830eddf0c1213759f87045369b862d2a74a5c1d89515912f84819a1fdf`
- Audit prompt-template SHA-256: `563ef78fe6f647c8533b3ec3058e0a970f171aa3da846b7eddac2035e04017ad`
- Audit runner SHA-256: `e49bc61ca90ad421f8e4dcfe42c3f3ee6bc52746c0105241646fb0bf3b919b0c`
- Per-event audit artifact SHA-256: `0eb814c56236e22d9c58b9c31a99e26fe6a793c9cdef9abfc449212803431d5e`
- Audit manifest SHA-256: `e41ab26f6d62be97d7d7588999a135c99b558103c8cc7d15372eebc5bbf811b4`
