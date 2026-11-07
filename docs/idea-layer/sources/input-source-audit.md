# Independent input-source search and audit

**Purpose:** find additional dated inputs for predicting ideas before a target speech. This is separate from transcript outcomes and semantic references. The active classifier remains headline-only. A separate retrospective 2019 source ablation is now documented in [`2019-full-year-preliminary-study.md`](../stories/2019-full-year-preliminary-study.md); it is a development diagnostic, not an independent accuracy estimate.

## The actual bottleneck

Roll Call Factba.se transcripts are already local: 1,990 speech records across 1988–2026. Only 38 have the retrospective seven-day headline snapshots used in the current experiment, and those snapshots were all retrieved after their target speeches. The slow work has been transcript-first, assistant-authored outcome inventories—not obtaining speech transcripts. Additional news or speaker-post archives enrich prediction inputs; they do not provide objective labels for ideas in a speech. Publication dates alone do not prove contemporaneous availability.

## Useful dated news-headline corpora found

### ABC News Australia: *A Million News Headlines*

- Kaggle source: <https://www.kaggle.com/datasets/therohk/million-headlines>; Hugging Face mirror: <https://huggingface.co/datasets/rajistics/million-headlines>.
- The pinned mirror has **1,226,258** rows with `publish_date` and `headline_text`, from **2003-02-19 through 2020-12-31**. Kaggle and the mirror declare CC0. The Kaggle description claims a 2021 end date, but the actual file ends in 2020.
- It supplies dated headlines for **11/38** target-event weeks from 2016–2020: **676–1,278 headlines per week**, 10,328 rows across the extracted windows. It is one Australian outlet, not a representative U.S. press corpus; dates have day precision and no article URLs. Although the dataset declares CC0, the uploader’s rights in the original publisher content are not independently established; keep the copy local and do not redistribute.
- Raw file and extracted windows are Git-ignored under `data/source-archives/abc-million-headlines-2003-2021/`. Raw SHA-256: `9ebbd3ce31413bbbfc467af291c2a08f3579cd84b462c495a688d6210e172212`; window snapshot SHA-256: `df7ebcaf8146e250eb319410832eb7da1e13ffa154384201b894318216a84426`.

### HuffPost News Category Dataset

- Kaggle source: <https://www.kaggle.com/datasets/rmisra/news-category-dataset>; mirror: <https://huggingface.co/datasets/khalidalt/HuffPost>.
- **200,853** headline/short-description records, actually dated **2012-01-28–2018-05-26**. Six target-event weeks have 427–632 distinct-URL articles each (3,486 total after per-window URL deduplication). The full corpus has 41 duplicate URLs and six blank headlines. This is another single-outlet sample and does not reach later events.
- There is a license discrepancy: Kaggle reports **CC BY 4.0**, while the Hugging Face mirror card says CC0. The original publisher’s content rights are not established by either dataset card; preserve attribution and do not redistribute. The local research copy and windows are Git-ignored under `data/source-archives/huffpost-news-category-2012-2018-v2/`.

These are useful additional **input** corpora, not new speech outcomes. Their publication dates and our retrieval now do not prove historical point-in-time availability. The 2019-focused use of ABC headlines is reported separately in [`2019-full-year-preliminary-study.md`](../stories/2019-full-year-preliminary-study.md).

## Speaker-post archives found independently

These should be tested as a distinct input modality, not silently merged into the headline-only news baseline.

### Historical Twitter archive

The [MarkHershey archive](https://github.com/MarkHershey/CompleteTrumpTweetsArchive), pinned at `8be27c439d0c8236d0de1c8b1b3fa54c5f8fc976`, has 54,324 posts from 2009–2021. Its earlier preceding-week extraction covered 11 events with 33–206 posts each (1,123 rows across those windows). A separate 2019 study filtered the full archive to 7,248 raw posts across 364 days; two duplicate 2019 permalinks were removed for the post-text ablation. Deduplicate by `Tweet URL`: the in-office CSV has 10 duplicate permalink rows overall, and its `ID` field is the account handle. `Time` lacks a timezone; the README says the archive includes posts deleted later without deletion dates. The repository declares MIT, but that does not establish reuse rights to underlying posts. Do not claim that every archived post was publicly available at the target time. Raw bytes and extracted windows remain Git-ignored under `data/source-archives/markhershey-complete-trump-tweets-8be27c439d0c/`.

### Matt Stiles archive and current CNN-hosted feed

- Upstream: <https://github.com/stiles/trump-truth-social-archive>; the README points to the live JSON feed <https://ix.cnn.io/data/truth-social/truth_archive.json> and says it is updated every five minutes. The README also says its former GitHub Actions workflow was disabled in October 2025.
- A single feed response retrieved **2026-11-05** had a `Last-Modified` time of `2026-11-05T01:39:57Z`, **37,951** records with UTC `created_at` timestamps, spanning 2022-02-14–2026-11-04. It yielded **25/38** target-event windows and 3,696 posts there (2,986 with nonempty text). Combined with the historical Twitter archive, these social sources cover **36/38** events; uncovered events are 1988-06-05 and 2022-01-29.
- The archive declares CC0, but that does not settle rights in the underlying posts or media. Its records have HTML content, 7,639 blank-text rows, media links, and current engagement counts. It lacks a reliable original/repost type field. Engagement counts were excluded from the extracted windows because they are current-at-retrieval, not historical features.
- The raw response and calendar-day-filtered windows are preserved locally, Git-ignored, under `data/source-archives/stiles-truth-social-cnn-feed-2026-11-05T0139Z/`. Raw SHA-256: `7a92c972003a68c75cbd1f96ef84e95582943c71cd5c07189a99f4bd9fb17af6`; extracted-window SHA-256: `4f71f2a9c1260aad87660e59e47c2d9bfcdf6f8e345745911b3cf73b652873a8`.
- These are retrospective archive observations. Creation timestamps do not establish that a post remained publicly visible before the speech; deletion history is unavailable. All extracted windows exclude the target date and later dates. No post text has been evaluated against a forecast.

### UCSB American Presidency Project Truth Social catalog

- Catalog: <https://www.presidency.ucsb.edu/documents/app-attributes/truth-social>. An initial metadata listing showed 734 documents: 700 Trump-account daily pages and 34 separate White House pages. Trump-account pages observed span 2024-12-01–2026-11-03; 19 target-event windows had 5–7 of seven daily pages. A page is not a post count, and White House posts must not be conflated with Trump-authored posts.
- Collection is paused: the site's `robots.txt` disallows `GPTBot` and `ChatGPT-User` and specifies a 10-second crawl delay. No UCSB post corpus was retained. Do not automate collection unless UCSB permits it or the user supplies an authorized export.

Other archives surfaced in the search:

- Data Driven Decision Lab lists a CC0 Kaggle CSV covering Twitter/X and Truth Social, 2009–December 2025, and links a 2026 resource page. The Kaggle metadata lists one 31.7 MB file, but an unauthenticated download probe returned 404 and the resource page exposed no static download link. I did not obtain or use it.
- A Hugging Face archive by Cameron Keplinger reports 38,045 Truth Social posts through 2026-11-03 and labels original, quote, and repost types. Its notice explicitly says the dataset does **not** grant rights to reuse the underlying posts or media, so it was not downloaded or used.

## Broader news sources and archives assessed

- **All the News 2.1**: the pinned Hugging Face mirror has 2,688,878 rows from 27 listed U.S. publications, mostly 2016–early 2020. A full local copy was acquired at dataset revision `a1f7e19b50048259d6006324c8b12ee4a0284ac5` (converted-Parquet revision `ef618cc31d5892f4e34615ef2e881c177819f7d4`). It has 655,456 dated 2019 rows across January–December, with 25 distinct nonempty publication values. Date/title/URL/publication were used in the separate 2019 study; article text was not a model feature. The mirror does not declare a clear reuse license; data and derived text remain local/Git-ignored. <https://huggingface.co/datasets/rjac/all-the-news-2-1-Component-one>
- **CC-News**: the pinned Hugging Face mirror has 708,241 rows and its card claims January 2017–December 2019 coverage. The retained local snapshot has only two rows with 2019 publication dates (77,598 rows have blank dates); it produced only eight overlapping 2019 event-window rows and was excluded from model inputs. The card says it is a small portion of the English Common Crawl news set; no clear reuse license is declared. <https://huggingface.co/datasets/vblagoje/cc_news>
- **RealNews**: the pinned Hugging Face dataset revision `e4ebb4e796a27cb7f32c23c6c9cb738d8fc78489` reports 32,797,763 rows (79.3 GB Parquet); its converted-Parquet revision is `46f95222dc6f36f19a5969f3ce5ff219b182379d`. A small remote field probe found `publish_date` strings such as `MM-DD-YYYY` and inconsistent `warc_date` serializations, potentially useful as capture-time evidence. A full event-window scan was attempted but stopped when the Hugging Face resolver returned HTTP 429 while opening shard `0009` (3,000-request/5-minute rate limit); no full corpus or event-window result was retained, and no coverage conclusion was drawn. The dataset card declares no license. <https://huggingface.co/datasets/Hieuman/realnews>
- **Common Crawl**: free public WARC/URL-index data has 129 listed indexes spanning 2008–2026. It can help verify that a *known URL* was captured before an event, but it is not a topic-searchable headline corpus; article discovery would need another source. <https://commoncrawl.org/get-started> · <https://index.commoncrawl.org/>
- **GDELT**: the project publishes large event/GKG archives and BigQuery access; GKG can supply dated URLs/entities/themes, not a ready full-text headline corpus. Two DOC API date-range probes returned HTTP 429, and I stopped without retrying. <https://www.gdeltproject.org/data.html>
- **New York Times Article Search API**: a plausible date-filtered, single-publisher archive. The unauthenticated endpoint returned HTTP 401; an API key is needed before testing coverage. <https://developer.nytimes.com/docs/articlesearch-product/1/overview>
- **News API** documents `from`/`to` search and published timestamps, but says the oldest available date depends on the plan; it is not a proven long-range archive. <https://newsapi.org/docs/endpoints/everything>
- **Internet Archive TV News** advertises closed-caption search across more than three million U.S. broadcasts. The search app is hosted on a subdomain whose `robots.txt` disallows crawling, so I did not automate caption search or download transcripts. <https://archive.org/details/tv>
- **Federal Register API** is accessible for date-filtered official policy documents, not general news. Metadata-only probes returned 633 documents for 2019-03-13–19, 49 for 2025-10-22–28, and zero for 1988-06-05–11; routine notices make the unfiltered counts noisy. <https://www.federalregister.gov/developers/documentation/api/v1>
- **Older news**: Reuters-21578 at UCI is a 1987 corpus and does not cover the 1988-06-05 target week. LDC's TIPSTER Volume 2 does include AP Newswire from 1988, but requires a license/agreement. LDC says its New York Times Annotated Corpus is no longer available. <https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection> · <https://catalog.ldc.upenn.edu/LDC93T3C> · <https://catalog.ldc.upenn.edu/LDC2008T19>

## Next experiment and limits

1. Keep the active headline-only classifier and its report unchanged. Treat the 2019 ablation as a separate development experiment; its strongest early ranking profile was All the News alone, while binary loss remained worse than the no-context prevalence reference.
2. Diagnose the 2019 per-event misses and phrase-label coverage before further tuning. Do not reuse this inspected 2019 slice as an untouched test set. Any future post-source experiment should preserve the separate modality, date cutoffs, deleted-post uncertainty, and the archive's missing timezone.
3. For the 1988 gap, the concrete identified route is licensed AP 1988/TIPSTER data. The 2022-01-29 event and a broader multi-outlet historical news archive still need more research; a keyed NYT API or institutional Factiva/Nexis/ProQuest access are possible routes to investigate. Internet Archive TV was not searched.

Raw corpora, derived event windows, and reports remain local and Git-ignored; none were committed. A separate research script and the methodological summary are tracked, but the active classifier and its headline-only result remain unchanged. Assistant-reviewed outcome inventories remain interpretive references, not objective semantic gold. All 2019 numerical scores are retrospective diagnostics against a weak automatic phrase extractor, not semantic accuracy.
