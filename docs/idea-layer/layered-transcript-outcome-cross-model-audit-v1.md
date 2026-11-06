# Layered transcript-outcome cross-model audit v1

## Purpose and interpretation

This is a bounded, transcript-only diagnostic of the layered outcome representation (mentions, abstract ideas, typed links). It is **not** a semantic-accuracy estimate, an independent forecast evaluation, or a quality gate. The target transcripts are labels; none of these audit outputs were used as prediction features or to score the frozen seven-day baseline. No pre-event news/context was supplied, so actual idea forecastability was deliberately **not scored**.

The auditor assigned separate 0–4 judgments for mention coverage, idea coverage, idea grounding, idea atomicity, and link quality. It also reviewed candidate ideas individually where feasible and returned cited issue/omission examples. These scores are subjective model judgments without a reference inventory. A different model is not ground truth, and exact-quote checks establish only that the cited text occurs in the cited turn—not that it semantically entails a candidate idea.

## Protocol and provenance

- The versioned audit implementation is `research/audit_layered_transcript_outcomes.py`; its prompt explicitly forbids outside knowledge and excludes event IDs/dates from the model input. The audit uses only the complete target-speaker transcript and the corresponding layered outcome row.
- It validates event-record and target-transcript hashes, validates the layered schema, blinds turn IDs, and requires exact transcript substrings for issue and missing-content examples. Candidate idea reviews are checked against the requested IDs. This is mechanical provenance/format validation, not semantic validation.
- The final auditor was `openai-codex / gpt-6.1-sol`; aliases are mutable. The 2019 candidate rows came from both `openai-codex / gpt-6-luna` (same provider, different model) and `deepseek / deepseek-v4-pro` (cross-provider). Thus 9 of 14 audited events are same-provider cross-model diagnostics and only 5 are cross-provider. The small cross-provider subset is not enough to establish independent agreement.
- Selection: eight OpenAI-extracted rows spaced across transcript-length rank among the 44 such rows, all five DeepSeek-extracted rows, and the separate long-speech smoke-test event. The resulting 14 events cover 32,530 target-speaker words, 635 mentions, 277 ideas, and 657 typed links. All 277 ideas were reviewed except in the long-speech smoke test, where 80 of 136 were sampled across prominence/forecastability strata. Selection is development-oriented, not random or representative.
- All 14 audit requests returned structurally valid responses. Across them, the auditor supplied 48 cited issue examples and 10 cited missing-content examples; each supplied quote was mechanically verified against its blinded source turn. The model also produced 221 individual idea reviews. Issue examples are capped, model-selected, and non-exhaustive; their counts are not error rates.
- A preliminary request through the configured DeepSeek endpoint was rejected for insufficient account balance and yielded no ratings. The completed run used the model above. Full transcript text and transcript-derived candidate layers were sent to the configured inference provider for the completed audit; raw audit output and source artifacts remain local and Git-ignored.

## Event-level judgments

The table reports the auditor's separate 0–4 ratings (higher is more favorable). These are not accuracy scores. `M cov.` = mention coverage; `I cov.` = idea coverage; `Ground.` = idea grounding; `Atomic.` = idea atomicity; `Links` = typed-link quality. `Reviewed` is individual idea reviews / candidate ideas.

| Event ID | Date | Extractor relation | Reviewed | M cov. | I cov. | Ground. | Atomic. | Links |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `dt45-19-a0977a0c` | 2019-03-23 | same provider | 5/5 | 4 | 4 | 4 | 3 | 3 |
| `dt45-19-b30091bf` | 2019-05-09 | same provider | 11/11 | 4 | 4 | 3 | 4 | 4 |
| `dt45-19-d00323a4` | 2019-05-22 | same provider | 8/8 | 4 | 4 | 3 | 3 | 2 |
| `dt45-19-a436e2b3` | 2019-05-26 | same provider | 1/1 | 4 | 4 | 4 | 4 | 4 |
| `dt45-19-6f306a44` | 2019-06-30 | same provider | 3/3 | 4 | 4 | 4 | 4 | 3 |
| `dt45-19-a1444f77` | 2019-07-23 | same provider | 4/4 | 4 | 3 | 4 | 4 | 4 |
| `dt45-19-ed9fee63` | 2019-08-07 | cross-provider | 11/11 | 4 | 4 | 3 | 3 | 3 |
| `dt45-19-1f61bbed` | 2019-10-08 | cross-provider | 34/34 | 4 | 4 | 3 | 3 | 3 |
| `dt45-19-18ebde99` | 2019-10-30 | cross-provider | 10/10 | 4 | 4 | 3 | 3 | 3 |
| `dt45-19-1ed603e3` | 2019-11-04 | same provider | 13/13 | 4 | 3 | 4 | 3 | 2 |
| `dt45-19-1143aee4` | 2019-11-07 | cross-provider | 16/16 | 4 | 4 | 4 | 3 | 3 |
| `dt45-19-81071469` | 2019-11-25 | same provider | 8/8 | 4 | 4 | 4 | 4 | 4 |
| `dt45-19-33210f4d` | 2019-12-18 | same provider; long smoke test | 80/136 | 4 | 3 | 3 | 2 | 2 |
| `dt45-19-9820d08f` | 2019-12-19 | cross-provider | 17/17 | 4 | 4 | 4 | 3 | 3 |

The event-macro means were 4.00 for mention coverage, 3.79 for idea coverage, 3.57 for grounding, 3.29 for atomicity, and 3.07 for link quality. The uniform maximum mention-coverage rating is a potential ceiling/bias signal, especially since the same audit returned omissions; it must not be read as proof of complete coverage. The five-event cross-provider subset had means of 4.00, 4.00, 3.40, 3.00, and 3.00 on those dimensions respectively. The different sample composition and very small `n` make subgroup comparison uninformative.

## Recurring diagnostic findings

The auditor most often questioned the *type* of mention-to-idea relation: content classified as `supports` was sometimes judged to directly express, elaborate, or merely contextualize the idea. It also flagged overstatement or attribution drift, conflation of distinct subjects, duplicate/umbrella ideas, overly narrow evidence spans, and omitted independent event-specific ideas. The long-speech smoke test had the lowest atomicity/link ratings; the auditor noted duplicated umbrella concepts, a speaker-attribution problem, and omissions. These are model-generated hypotheses for further investigation, not adjudicated defects.

Across the 221 individually reviewed ideas, the auditor emitted 50 non-exclusive issue flags: 15 `conflated`, 12 `duplicate`, 7 `overstated`, 6 `over_split`, 4 `vague`, 3 `wrong_link`, 2 `missing_direct_expression`, and 1 `understated`; it emitted no `unsupported` flags at the individual-review level. Separately, among the 48 cited issue examples, 24 were categorized `wrong_link`. Ten missing-content examples appeared across four events. These different counts arise from distinct response fields and caps; neither is a corpus-wide count or an estimated error rate.

Per-event examples (paraphrased; no source quotations reproduced here):

- **2019-03-23:** generic praise was judged somewhat over-split, with a relation-type mismatch.
- **2019-05-09:** some evidence spans were considered too narrow; the auditor identified omissions in a personal anecdote and a game recap.
- **2019-05-22:** an account was judged conflated/overstated, and generic praise was sometimes linked as though it directly expressed a narrative proposition.
- **2019-05-26:** a mention was judged to infer a signing detail not explicit in its cited transcript text.
- **2019-06-30:** one relation was judged too weak for a directly stated meeting.
- **2019-07-23:** the vote outcome was judged underrepresented at the idea level, and an invitation/attendance subject was marked missing.
- **2019-08-07 (cross-provider):** the auditor questioned converting questions into assertions and adding specificity not established by the transcript.
- **2019-10-08 (cross-provider):** it flagged some career-episode grouping and relation choices, alongside a possibly overstated superlative.
- **2019-10-30 (cross-provider):** family history and continuing service were judged conflated; one account's objective was considered narrowed.
- **2019-11-04:** an umbrella idea was judged to combine distinct event-specific subjects; several subjects were marked missing.
- **2019-11-07 (cross-provider):** the auditor questioned a causal qualification and several contextual details typed as support.
- **2019-11-25:** no major issue was flagged, but this is only one model's judgment.
- **2019-12-18 (long smoke test):** duplicate/umbrella ideas, a possible speaker-attribution error, weak links, and three omitted subjects were flagged; only 80/136 ideas received individual reviews.
- **2019-12-19 (cross-provider):** one paraphrase was judged unsupported/reversed, with policy bundling and relation-type concerns.

## Implications and limits

This audit supports only a cautious development conclusion: the layered design appears useful for separating transcript-local mentions from idea targets, but the extracted representation still has notable grouping and relation-typing risks. The outputs should not be promoted to semantic gold, used to train a forecast model as if verified, or used to claim forecastability. The current findings do not define a credible semantic quality gate because no independent reference labels or human review are available. The 2019 events and chunked smoke test were already inspected development material; this is not independent evaluation.

The audit prompt, candidate artifacts, and provider/model aliases are mutable/subjective factors. A different model may share biases or miss the same content. Exact citations validate location only, and the model's completeness scores cannot prove recall. The audit evaluates transcript representation, not whether any idea was predictable from contemporaneous news. It also says nothing about point-in-time source availability or the seven-day baseline's forecasting performance.

The first DeepSeek attempt is preserved as a local failed artifact (`data/source-archives/2019-layered-outcome-cross-model-audit-v1.jsonl`, 13 model errors and no ratings); the completed run is `data/source-archives/2019-layered-outcome-cross-model-audit-v2.jsonl`. Completed audit output SHA-256: `1202ff7fca5bfa91165f0eb42207a5177cca9480224223e8a41202d2be639ace`; manifest SHA-256: `6ce9561b21fca6eddd2d945b95f897f076f31aeaf981d23251a73cb857808698`. The source hashes and all per-event judgments are recorded in the local manifest/JSONL. No transcript-derived records, quotes, audit outputs, or checkpoints are committed.

## Next steps

1. Keep the seven-day classifier frozen and keep all layered outcomes out of active prediction inputs.
2. Inspect the identified relation-typing, attribution, overstatement, and omission cases as development diagnostics; do not tune to the 2019 scores and call them independent validation.
3. Test the chunked extractor on more long transcripts and audit boundary/consolidation behavior before broad use. A larger cross-provider audit would require an available provider budget and remains another model diagnostic, not a substitute for independent semantic reference labels.
4. Do not claim a semantic quality gate or forecastability until the project can define a credible reference/evaluation protocol under its no-human-review constraint.
