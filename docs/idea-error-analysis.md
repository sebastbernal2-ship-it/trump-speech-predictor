# Idea-model error analysis (development diagnostics)

This is an assistant-authored diagnosis of the frozen retrospective experiment, not objective semantic adjudication. The target phrase extractor remains the scored reference. Predictions that do not match an extracted phrase may still express a related idea; conversely, exact phrase matches do not establish substantive coverage.

## What was evaluated

The 38-snapshot union has 32 events with at least one extracted phrase. Chronological grouping by event date produced **25 train, 3 validation, and 4 development events**. The development events contain 45 extracted target IDs total. The selected model is Bernoulli Naive Bayes with `alpha=4`, `min_df=3`, 20 recent distinct headlines, stable captured order for publication-date ties, and logit temperature 8.0. Selection searched 72 configurations using only three validation events.

On the four development events:

| Measure | Recent-headline model | No-context prevalence reference |
|---|---:|---:|
| Binary log loss | 0.708 | 0.562 |
| nDCG@10 (macro) | 0.475 | 0.439 |
| P@4 (macro) | 0.563 | 0.563 |
| Recall@10 (macro) | 0.371 | 0.315 |

These are mixed: context modestly improves nDCG and recall, ties P@4, and worsens binary loss. They do not establish a useful model. The validation slice itself had P@4 0.083, recall@10 0.056, and nDCG@10 0.101 under the selected configuration; the apparent development ranking lift is unstable and selection-sensitive.

## Where the errors occur

| Target event | Extracted targets | Top-10 exact-ID hits | Unseen before event | Seen targets missed at top 10 |
|---|---:|---:|---:|---:|
| 2025-10-29 | 10 | 3 | 1 (`artificial intelligence`) | 6 |
| 2025-12-19 | 17 | 5 | 1 (`electricity price`) | 11 |
| 2026-07-28 | 9 | 4 | 0 | 5 |
| 2026-11-02 | 9 | 4 | 1 (`iran war`) | 4 |
| **Total** | **45** | **16** | **3** | **26** |

Thus most top-10 misses (26 of 29) were labels the fitted classifier could in principle rank, but did not rank high enough. Three others were impossible for this closed-label model: its candidate set contains only phrase IDs observed in earlier paired examples. For example, the 2026-11-02 transcript has the `iran war` label, and the preceding-week snapshot contains many headlines explicitly mentioning Iran and war, but that label had not appeared in the model's earlier paired training examples. The model cannot propose it, however strong the context signal.

The full 32-example set has 87 distinct phrase IDs; 32 occur once and 45 occur at most twice. This is severe label sparsity for a context-conditioned classifier. It must learn headline-to-label relations for dozens of labels from 25 training events, often with only one or two examples.

## Concrete signal and representation failures

- **Lexical features do not understand paraphrase or named legislation.** For the 2026-07-28 event, a selected headline says “SAVE America Act” (the third recent source in the frozen snapshot), while the transcript extractor labels `id voter`. The bag-of-words model has no learned semantic bridge from the act name to voter ID. The target transcript also discusses Iran, a nuclear weapon, and ending the war in turns `dt47-26-f641fd3a:turn:0006`, `:turn:0007`, and `:turn:0011`; the compact pair extractor captures “nuclear weapon” but not the broader Iran-war idea in this event. The first is a model representation miss; the latter is a target-label granularity miss.
- **A direct news phrase can still be ranked poorly.** Before the 2025-10-29 event, a headline titled “What's Next for US-Russia Relations and the War in Ukraine?” appears within the selected 20-title set. The target `russia ukraine` is nevertheless absent from the top 10. This is evidence of a weak learned context-to-label ranking relationship, not merely absent input evidence.
- **A hard recent-headline cap can discard older, relevant evidence.** For 2026-07-28, a headline about Iran's Pickaxe Mountain and an underground nuclear site is in the seven-day candidate pool but well beyond the selected 20-title cap. The cap was selected on only three validation events. It trades noise for recall and may discard useful older-in-week signals.
- **Label IDs are lexical pairs, not a semantic ontology.** IDs such as `east middle` are sorted token pairs; output now includes a readable training-derived surface (`Middle East`), but this changes display only. It does not repair incomplete labels, consolidate semantic variants, or decide whether an idea was substantively addressed.

## Semantic-reference coverage check

Before this expansion, 20 local assistant-reviewed outcome records overlapped the 38 snapshot events: 18 transcript-first development inventories (blind items 003–020) and two explicitly post-hoc, nonblind records. Among the 18 transcript-first inventories, 13 cover the full or complete available transcript/excerpt, one is a very brief diplomatic statement, and four are low-information, fragmentary, or completeness-uncertain. The two post-hoc records are not blind references. These are interpretive references, not objective gold, and their existence does not make retrospective news snapshots contemporaneously available.

This expansion adds two transcript-first inventories, bringing the referenced event count to 22 of 38. The 2024-10-21 campaign speech covers all 165 available target-speaker turns and records 16 cited ideas; the 2025-04-16 Easter address covers all 22 available target-speaker turns and records 7. For both, event metadata was visible during event selection, so they are not fully blind. Their event-specific forecasts, headline content, and weak phrase labels were not opened before the outcome inventories were frozen. They can inform training for strictly later events, but should not be treated as independent test references. The local Git-ignored artifacts are `data/rollcall-assistant-reviewed-outcomes-expansion-item021-v1.jsonl` (SHA-256 `dfc7e1b22ecf2f49283b58b0a6e6706a1c0d49af57c9185235d5179341629de2`) and `data/rollcall-assistant-reviewed-outcomes-expansion-item022-v1.jsonl` (SHA-256 `a5f8bf2ff9fb60ff3460531aa13c8619de5dba0e14e1911f2ab3f78871aac540`). Citation/structure checks passed. No model predictions were evaluated against these new references.

A second expansion adds three more transcript-first inventories: 1988-06-05 (49 turns, 5 ideas), 2017-06-16 (44 turns, 5 ideas), and 2020-07-04 (45 turns, 9 ideas). This brings the number of referenced snapshot events to 25 of 38. As with the prior expansion, event metadata was visible during selection, while event-specific forecasts, headline content, and weak phrase labels were not opened before each inventory was frozen. The assistant read the complete available target-speaker transcripts and validated citations against source turns. Local Git-ignored artifacts: `data/rollcall-assistant-reviewed-outcomes-expansion-item023-v1.jsonl` (SHA-256 `bc0ae03b21f0bd7b18272f8ea7617bec3b30f3d968a6b196497cdf291d8b6aed`), `data/rollcall-assistant-reviewed-outcomes-expansion-item024-v1.jsonl` (SHA-256 `52011a86a9f9e0bb56812e4641c65bcc0d02e60fb043a5c556b136e996233053`), and `data/rollcall-assistant-reviewed-outcomes-expansion-item025-v1.jsonl` (SHA-256 `135e28dc19bfd0d8aa681daa38462213e29802f0a14d3da7b99ee8591a4ae05e`). No model predictions were evaluated against these references.

A third expansion adds two 2016 campaign-speech inventories: 2016-09-08 (79 available turns, 9 ideas) and 2016-11-05 (92 available turns, 14 ideas), bringing coverage to 27 of 38 snapshot events. Both transcripts contain marked missing/technical-difficulty spans; these are documented in the local references. Event metadata was visible at selection, but event-specific forecasts, headlines, and weak labels were not opened before the inventories were frozen. The local Git-ignored artifacts are `data/rollcall-assistant-reviewed-outcomes-expansion-item026-v1.jsonl` (SHA-256 `d4c7bb6a675a3062d022b1bd9f4e47c9cfddfc27f3c2b7feebc64d50a7d70227`) and `data/rollcall-assistant-reviewed-outcomes-expansion-item027-v2.jsonl` (SHA-256 `2fd73437baf2372b7b3380aedd87eacd2a117c0914750b4a53c4f991ad0b35ef`). Item 027 v2 corrects its coverage metadata to record two technical-difficulty markers; ideas and citations are unchanged from v1. Citation/structure and source/transcript hash checks passed. These new references have not been used for model evaluation or retraining.

One of the four development targets, 2025-10-29, has a full-transcript-only inventory frozen before forecasts, headlines, and weak labels were opened. It lists 18 forecastable semantic ideas, while the compact extractor produces 10 lexical-pair targets for that event. Those counts are not directly comparable: the reference separates ideas such as tariffs, trade leverage, and deficit effects, whereas the extractor can split, merge, or capture incidental phrases. In particular, the extractor marks `war world` positive because the transcript uses World War II as a historical shipbuilding comparison; the frozen semantic inventory does not treat that as a substantive World War idea.

A later assistant comparison of the model's top 10 against that frozen reference is preserved locally in `data/rollcall-idea-semantic-forecast-match-dev-item013-v2.jsonl` (Git-ignored; SHA-256 `25998ee43f7b8570bb4bbf9bb43680b58ec43d712f884510ec8a40812eaaab19`). It is explicitly post-hoc and diagnostic, not a blind evaluation: the candidate matcher had already seen the forecast/news evidence. The assistant judged two top-10 items as clear semantic matches (Trade deals; American energy), one partial (Middle East), two uncertain (World War; Nuclear weapons), and five unsupported by the reference. This contrasts with the three exact weak-label hits (Middle East, World War, Trade deals): it exposes both a likely phrase-label false positive (World War) and a semantic match (`American energy`) that exact-label scoring misses. The same comparison identifies a wrong-entity error: the model predicts North Korea, while the transcript's substantive Korean topic is the U.S.–South Korea alliance. These are one assistant's post-hoc judgments on one event, not accuracy estimates.

## What appears to work—and what does not

The model often ranks recurring, well-supported phrase IDs such as Middle East, Open borders, Second Amendment, and Republican Party. That yields modest average ranking lift on four events. But the same handful of recurrent labels crowd the top 10, while event-specific proposals and policy details are missed. The prevalence baseline ties P@4, and the model's binary loss is worse; the model has not shown that it converts current-news evidence into reliable idea-level likelihoods.

A post-hoc input-union check that round-robined topic-family and speaker-name query results for the same 20 events did not improve development metrics (nDCG@10 0.447 and loss 0.710 versus 0.475 and 0.708 for the selected input). This is not an independent comparison: it uses the same inspected four-event slice. Simply adding more retrospectively retrieved headlines is not demonstrated to solve the problem. Local ignored artifacts: snapshot union `data/rollcall-context-snapshots-query-union-v1.jsonl` (SHA-256 `fa95218cb51206d639d079e92ce592ce8ba1f5f814f8e9bedfb69654b04337ce`) and report `data/rollcall-context-to-idea-evaluation-query-union-v1.json` (SHA-256 `70e812e94ed2d64113a70345beb6a1644ec2bcbb770593ed3e10e6fd2855353a`).

## What to change

### High level: fix the supervised task signal and its scale

1. Define an “idea” as a semantic proposition/topic addressed in a speech, with criteria distinguishing a passing mention from substantive treatment. The current two-word lexical-pair outcome is not adequate as the sole target for that task.
2. Produce assistant-only, transcript-first semantic inventories for paired events, with exact turn citations, canonical IDs, uncertainty, and an explicit completeness assessment. Preserve them as interpretive references, not semantic gold. Do not inspect forecasts/news while first building an event's outcome inventory.
3. Expand the number of paired context/transcript events substantially. With 25 training pairs and 87 labels, tuning another small classifier grid is not a plausible route to robust performance. Keep retrospective retrieval provenance and timing caveats explicit; do not pretend publication dates prove contemporaneous availability.

### Mid level: make prediction open-vocabulary and context-sensitive

1. Generate candidate ideas from the pre-event context as well as from prior supervised labels, then rank both candidate sources. A closed list of historically observed phrase IDs makes genuinely new event-specific ideas impossible to predict.
2. Compare lexical NB with a semantic headline/idea representation (for example, pretrained text embeddings plus a small supervised ranker) only after target concepts are stable and the paired dataset is larger. Preserve cited headlines as evidence and allow abstention/uncertainty.
3. Retain a no-context prevalence baseline and use rolling chronological folds with enough events to estimate both ranking and probability loss. Do not choose a model on a three-event validation slice or claim four-event development lift as independent accuracy.

### Low level: improve the current experiment without hiding its limits

1. Keep headline titles deduplicated, strip publisher suffixes, preserve capture order for date ties, and test recent-only against week-stratified selection. Record which titles/features each forecast actually used. Do not assume a top-20 cap is intrinsically correct.
2. Report, for every event and idea, whether it was in the prior label vocabulary, its training support, its rank or miss, and the headlines offering direct or semantic evidence. Separate candidate-generation failures from ranking failures and weak-label misses.
3. Calibrate probabilities and choose output cardinality/abstention rules on validation data only, with more than a few events. Keep loss, ranking, coverage, and semantic reference agreement distinct.

The next high-value step is the **transcript-first target-definition/coverage audit plus expansion of paired context examples**, not another hyperparameter sweep and not sentence generation. The current code/model artifact and per-event predictions are in the ignored local report `data/rollcall-context-to-idea-evaluation-v7.json` (SHA-256 `5589eb462e6db386a972393822192b36f449ed88b5668d930442821391b5fd87`). The report is a small retrospective development diagnostic only.
