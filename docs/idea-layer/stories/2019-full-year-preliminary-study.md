# 2019 full-year preliminary context-to-ideas study

**Status:** retrospective development/backtest diagnostic. This is a separate source ablation; it does not replace or retrain the active headline-only baseline. The 2019 results have been inspected during development and are not independent, prospective, or semantic-accuracy estimates.

## Scope and target records

The local Roll Call Factba.se candidate corpus contains **191 Trump speech/remarks records dated 2019-01-08 through 2019-12-25**, on **147 distinct dates**. The compact transcript phrase extractor returns at least one target phrase for **142 records** (117 distinct dates); **49 records have no extracted outcome** and are excluded from metric denominators. These are the records in the local candidate corpus, not a guarantee that every 2019 Trump speech is represented. A separate transcript-only LLM pilot generated cited open-vocabulary candidate atoms for these 49 records; see [`transcript-idea-atoms-pilot.md`](../points/transcript-idea-atoms-pilot.md). It did not change this study's labels or metrics.

The model predicts transcript-extracted issue phrases from archived context dated in `[event_date - 7 days, event_date)`. Target transcripts, event titles, locations, target-day sources, and later sources are not features. The training set contains **641 labeled 2016–2018 events**; the 2016-01-07 record is excluded because the All the News archive begins on 2016-01-01 and cannot provide its full preceding week. All 2019 outcomes are held out from training and hyperparameter selection.

## Sources actually used

- **All the News 2.1** ([Hugging Face dataset](https://huggingface.co/datasets/rjac/all-the-news-2-1-Component-one)): pinned dataset revision `a1f7e19b50048259d6006324c8b12ee4a0284ac5`, converted-Parquet revision `ef618cc31d5892f4e34615ef2e881c177819f7d4`. The local snapshot contains **2,688,878 rows**; **655,456** have a 2019 date (Jan 1–Dec 31). Those rows have 25 distinct nonempty publication values, 12,577 missing URLs/publisher values, 642,877 distinct nonempty URLs, and 566,799 distinct titles. Date, title, URL, and publisher drive model selection/features. The v2 exact-body ablation additionally reads All the News article bodies only to compute normalized SHA-256 hashes for duplicate removal; raw body text is not a model feature and is not written to study outputs.
- **ABC News Australia: A Million News Headlines** ([Kaggle](https://www.kaggle.com/datasets/therohk/million-headlines), [Hugging Face mirror](https://huggingface.co/datasets/rajistics/million-headlines)): **34,060** dated 2019 headlines. The source has day precision and no per-article URLs; it is a single Australian outlet.
- **MarkHershey’s CompleteTrumpTweetsArchive** ([GitHub](https://github.com/MarkHershey/CompleteTrumpTweetsArchive)): **7,248** raw 2019 posts across 364 days; two duplicate permalink rows were removed, leaving 7,246 unique-URL rows for the input table. `Time` has no timezone. This is a separate post-text modality, not part of the headline-only baseline; deleted-post history is unavailable.
- **CC-News** ([Hugging Face dataset](https://huggingface.co/datasets/vblagoje/cc_news)): pinned dataset revision `81eb2ce0d2a9dad6ad16b68ef750ec290880fa36`, converted-Parquet revision `df7dd7315a14b46e495e527d517e1acebf3645a7`. The card claims coverage through December 2019, but the retained 708,241-row snapshot has only **two rows with a 2019 publication date** (and 77,598 blank dates overall). It contributes just eight overlapping event-window rows and is excluded from the models.
- HuffPost ends in 2018 and the Stiles/CNN Truth Social archive begins in 2022, so neither supplies 2019 inputs.

### Additional source probed but not used

**RealNews** ([Hugging Face](https://huggingface.co/datasets/Hieuman/realnews)), pinned at dataset revision `e4ebb4e796a27cb7f32c23c6c9cb738d8fc78489` and Parquet conversion revision `46f95222dc6f36f19a5969f3ce5ff219b182379d`, has 32,797,763 rows in 79.3 GB of converted Parquet. A small remote probe confirmed publication-date and WARC-capture fields, but an attempted event-window scan was stopped by an HTTP 429 resolver rate limit while opening shard `0009`; no full corpus or event-window output was retained, and no 2019 coverage conclusion is available. The dataset card declares no license. The All the News 2.1 and CC-News cards also do not establish clear reuse rights; both were retained and used locally at the user's request. Raw corpora and derived text remain Git-ignored and are not included in this repository change.

The separate [`story-graph-architecture.md`](story-graph-architecture.md) audit documents exact repeated-body counts and their limitations. The v2 ablation below uses those hashes only to collapse exact repeated bodies before selecting headlines; it does not implement semantic story clustering.

## Context-window coverage

Each figure below sums eligible source rows across all 191 target-event windows, so a story/post can recur in overlapping windows.

| Source | Events with ≥1 item | Rows across windows | Mean per event | Range per event |
|---|---:|---:|---:|---:|
| ABC headlines | 191/191 | 126,130 | 660 | 454–743 |
| All the News 2.1 | 191/191 | 2,460,436 | 12,882 | 7,327–23,696 |
| Trump Twitter archive | 191/191 | 27,537 | 144 | 45–287 |
| CC-News | 8/191 | 8 | 0.04 | 0–1 |

The classifier uses at most the **20 newest distinct items per source per event**, rather than all items in each window. Thus the ABC+All the News profile has up to 40 items, and the profile adding Twitter has up to 60; these are source-bundle comparisons, not equal-total-input-budget tests. In the v2 exact-body ablation, All the News copies are collapsed by normalized body hash (falling back to normalized title when body is missing) before the same title deduplication and 20-item cap. Zero same-day/future items were found across the 1,037 2016–2019 context snapshots generated for training and evaluation.

## Evaluation protocol

The Bernoulli Naive Bayes model uses unigram/bigram presence features, with the existing issue phrase extractor as weak target labels. For each profile, alpha, minimum feature document frequency, and logit temperature were selected using a chronological **2016–2018-only** validation split (519 earlier training events, 122 validation events). Each profile selected `alpha=0.5`, `min_df=3`, and temperature `128`; the item budget was fixed at 20 per source and selected the latest distinct items. The final model for each profile was then refit on all 641 labeled 2016–2018 events and scored on the 142 labeled 2019 records. No 2019 labels informed fitting or configuration choice.

The no-context reference ranks training-extracted phrases by their prevalence in the same 641 historical labels. Metrics below are macro-averages over 2019 speech records; lower binary log loss is better.

| Input profile | P@4 (model / prevalence) | Recall@10 (model / prevalence) | nDCG@10 (model / prevalence) | Binary log loss (model / prevalence) |
|---|---:|---:|---:|---:|
| ABC headlines | 0.275 / 0.222 | 0.345 / 0.287 | 0.345 / 0.320 | 0.255 / 0.191 |
| All the News 2.1 | **0.301 / 0.222** | **0.361 / 0.287** | **0.371 / 0.320** | 0.253 / 0.191 |
| ABC + All the News | 0.299 / 0.222 | 0.343 / 0.287 | 0.361 / 0.320 | **0.249 / 0.191** |
| Trump Twitter text | 0.280 / 0.222 | 0.358 / 0.287 | 0.356 / 0.320 | 0.252 / 0.191 |
| ABC + All the News + Twitter | 0.301 / 0.222 | 0.347 / 0.287 | 0.369 / 0.320 | 0.280 / 0.191 |

All profiles had mean candidate coverage of **0.975** of the extracted 2019 target phrase IDs (about **0.141 previously unseen target IDs per event**). For All the News alone, the model had zero top-10 matches on **34/142** labeled events; its nDCG@10 exceeded, tied, or trailed prevalence on 65, 37, and 40 events respectively. Its binary loss beat prevalence on only 23/142 events.

### Exact-body-dedup ablation

The v2 ablation keeps the same seven-day cutoff, 20-item/source cap, training labels, and classifier, changing only All the News candidate deduplication: exact normalized body hashes are collapsed within each event window before normalized-title deduplication and the cap. For a repeated body, the newest dated record is retained; rows without a body hash fall back to title deduplication. Article bodies are not input features. The same chronological tuning procedure was rerun per profile, and all profiles selected the same settings (`alpha=0.5`, `min_df=3`, temperature `128`).

| Input profile | P@4 | Recall@10 | nDCG@10 | Binary log loss |
|---|---:|---:|---:|---:|
| All the News, title dedup (v1) | 0.3011 | 0.3613 | 0.3710 | 0.2531 |
| All the News, exact-body dedup (v2) | 0.2993 | 0.3602 | 0.3706 | 0.2531 |
| ABC + All the News, title dedup (v1) | 0.2993 | 0.3425 | 0.3605 | 0.2493 |
| ABC + All the News, exact-body dedup (v2) | 0.2993 | 0.3428 | 0.3614 | 0.2493 |
| Prevalence reference | 0.2218 | 0.2866 | 0.3201 | 0.1908 |

These are macro-averages over the same 142 labeled 2019 events; lower binary log loss is better. The body-dedup change therefore made virtually no aggregate difference: All the News alone decreased slightly on the ranking metrics, while the two-source profile changed negligibly. Every profile still had worse binary loss than prevalence. Candidate coverage remained about 0.975. Distinct-date summaries and per-event comparisons are in the local JSON report.

Summing per-window counts across the 191 event windows (so overlapping windows can count the same row repeatedly), there were 2,342,449 nonempty-body rows and 2,167,737 per-window distinct body hashes, for a 174,712 repeated-row excess. These are not unique-corpus counts. Exact repeated bodies occurred among the selected top-20 All the News items in 12/191 windows (8/142 labeled events), changing the model feature document in those same 8 labeled events. The selected item count remained 20 in each profile. This explains why a broad reduction in the archive's duplicate rows had almost no effect on this classifier's already-capped inputs.

This is only an exact-content preprocessing ablation—not story/update extraction, a larger temporal context model, or the proposed graph model. Exact body equality can still reflect boilerplate or other non-story identity; paraphrases and substantive updates are not detected. The unchanged aggregate results do not establish that story-level representation has no value.

### Preliminary interpretation

The headline-conditioned model shows a modest ranking lift against this lexical target set, strongest for All the News alone. Adding ABC or archived Trump posts did not improve ranking over All the News alone under this protocol. **Every context profile still had worse average binary log loss than prevalence**, and the lift is not consistent event to event. This is not evidence of reliable semantic forecasting. The results are sensitive to the compact phrase ontology, model calibration, archive composition, and the chosen item budget. The prior v7 settings are also retained in the local report as a frozen diagnostic; their loss was substantially worse on these sources.

## Limitations and next step

- The target phrase extractor is incomplete and lexical, not objective or semantic gold; no human annotation was used. Forty-nine 2019 transcript records have no extracted outcome.
- The transcript corpus is a retrospective candidate subset. Events on the same date share context, and adjacent event windows overlap; event-level scores are correlated. The JSON report also provides distinct-date macro summaries.
- The news corpora and Twitter archive were retrieved after the events. Pre-event publication/creation dates do not prove the material was then visible, indexed, or available. All the News and Twitter date strings have no verified timezone; Twitter's archive also lacks timestamps for later deletions.
- Results were inspected during this development iteration, and seven source profiles (including two exact-body-dedup variants) were compared. Do not reuse 2019 as an untouched test set or present these numbers as prospective accuracy.
- The unknown/unclear reuse terms are a reason to keep the raw sources and text-derived artifacts local and uncommitted.

A further useful step is to diagnose the per-event misses and, separately, seek genuinely point-in-time news captures. Do not change the active headline-only baseline on the basis of this small retrospective slice.

## Local reproducibility artifacts

The source-specific processing/evaluation script is [`../../research/2019_context_study.py`](../../../research/2019_context_study.py). Run it with DuckDB installed; it writes a new v2 output directory and refuses to overwrite existing output:

```sh
uv run --with duckdb python research/2019_context_study.py
```

Raw sources and outputs are Git-ignored under `data/source-archives/`. The v1 report was generated by the script at commit `9010f8b`; the current script generates v2. Run manifests record pinned source revisions and hashes. Key v1 output SHA-256 values:

- `event-source-windows-v1.jsonl`: `8149e52f543180ee61d2ec41838f4822af6960e3ff004a473e8dd25408b29763`
- `2019-target-inventory-v1.json`: `d4ad34413a1e023912125afbcd7a6d81e35451646c97545e8e00ea3ba84b7b76`
- `2019-preliminary-model-evaluation-v1.json`: `f9f7e3c3601a27e9f96d1f57a34dfcd54078c327496a9a67d9903b40e124ffd8`
- v1 study script: `3175f15eaad31c409489f383b5a967a4910abe5ac2b3b6ef2ac3aafd339c5f51`

V2 exact-body-dedup run, under `data/source-archives/2019-full-year-context-study-v2-exact-body-dedup/`:

- `event-source-windows-v2.jsonl`: `39ae44914be510931fdbb57b3719c338607826991188b4b676555d9c85262dca`
- `2019-target-inventory-v2.json`: `2a1efab132d6d0fda08b6a8aa8221b1348af0b2d2a6cd5c764dd110eb1add8c0`
- `2019-exact-body-dedup-evaluation-v2.json`: `62867ced896b9ab92f1657ae84721029a2e607269019dc606c1e7375ef6e8490`
- V2 study script: `9752e5dc2696462801b4815c0bd39d76bc7c5a8230c49f89fab9c927f68188a9`
