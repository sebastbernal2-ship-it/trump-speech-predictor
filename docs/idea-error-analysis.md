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
