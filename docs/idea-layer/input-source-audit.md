# Additional input-source audit

**Purpose:** assess whether the named public sources can expand the pre-speech *input* data. This is separate from the transcript-outcome/semantic-reference audit. The active headline-only baseline has not been changed, and no forecast has been scored against these sources.

## What is actually taking time

Roll Call Factba.se transcripts are not the acquisition bottleneck: the local ignored `data/rollcall-by-type/trump-speech-candidates.jsonl` contains 1,990 speech records, while the retrospective headline experiment has only 38 event snapshots. The semantic inventory is still assistant-authored and transcript-based; tweet archives provide predictor-side context, not objective labels for speech ideas. They can make the input richer, but they do not eliminate outcome-review work.

## Roll Call Factba.se

The active corpus and its event-turn structure already come from Roll Call Factba.se; collection code is in `speechprefix/rollcall.py`. The speech corpus spans 1988–2026. Do not recrawl transcripts just to increase the label count: the paired seven-day context is what is missing for most corpus events. The current 38 headline snapshots were all retrieved after their events, so they remain retrospective evidence and do not prove what a forecaster could have seen then.

## MarkHershey `CompleteTrumpTweetsArchive`

- Source: <https://github.com/MarkHershey/CompleteTrumpTweetsArchive>, pinned source commit `8be27c439d0c8236d0de1c8b1b3fa54c5f8fc976`.
- Its README reports 31,249 pre-office posts (2009-05-04–2017-01-19) and 23,075 in-office posts (2017-01-20–2021-01-08). Those counts and ranges match the retrieved CSVs: **54,324 rows** total.
- The CSV fields are `ID`, `Time`, `Tweet URL`, and `Tweet Text`. `ID` is the account handle, not a unique post ID; deduplicate with the permalink. The in-office file has 10 duplicate permalink rows; the pre-office file has none.
- `Time` has minute precision but no timezone. The README also says the archive includes tweets deleted later, without giving deletion times or evidence they remained publicly visible before each speech. Therefore this archive is useful for a **retrospective speaker-post feature experiment**, but not as proven point-in-time/publicly available context. Do not claim prospective availability or silently assume a timezone/deletion history.
- A date-only, event-day-exclusive lookback (`event_date - 7 days <= post_date < event_date`) contains 33–206 archived posts for 11 of the 38 snapshot events (2016–2020). These are source-volume counts only; post text was not compared to outcomes.
- Retrieved source bytes, README, license, and a provenance/hash manifest are preserved locally under `data/source-archives/markhershey-complete-trump-tweets-8be27c439d0c/` and Git-ignored. CSV SHA-256: `94e98ef1320c3d7fe659e308e77b49587fbb6f8db28241fb6b80cfff756d39e6` (pre-office) and `5b35db527bff9bc730e0eb4d8947cb7229a14ef894d9f43abd7b1caf0f40e319` (in-office). The repository declares MIT, but verify that this grant applies to the underlying user-generated post text before redistributing it.

The existing `speechprefix/idea_layer/posts.py` normalizer expects a provenance-bearing export with timezone-aware publication timestamps and availability evidence. It cannot safely consume this CSV unchanged; the date/time ambiguity and later-deleted posts need explicit representation rather than fabricated timestamps.

## UCSB American Presidency Project Truth Social catalog

- User-provided catalog URL: <https://www.presidency.ucsb.edu/documents/app-attributes/truth-social>.
- An initial metadata-only listing check showed 734 documents: 700 Trump-account daily pages (`truth-social-posts-*`) and 34 distinct White House pages (`white-house-truth-social-posts-*`). The Trump-account pages observed span 2024-12-01–2026-11-03; 19 snapshot-event lookback windows had 5–7 of seven daily pages. These are page-coverage counts, not counts of individual posts.
- Keep the White House account separate from Trump-authored posts. A daily page may contain multiple original-post links, so one page must not be treated as one post.
- **Collection is paused.** The site's `robots.txt` disallows `GPTBot` and `ChatGPT-User` and specifies a 10-second crawl delay. I stopped the metadata check when I found this; no UCSB post corpus was retained. Do not automate further collection from this site unless UCSB provides permission or the user supplies an authorized export. Treat the counts above as a limited source-discovery observation, not a complete validated corpus.

## Other sources and safe next experiment

Historical news archives (for example, GDELT and Internet Archive captures) are candidates for input expansion, but a later search result or capture does not prove that the same story was available to a forecaster at the target time. Two low-volume GDELT DOC API date-range probes (2019-03-13–19 and 2025-10-22–28) returned HTTP 429; I stopped without retrying, so this gives no coverage conclusion. Audit source timestamps, coverage, reuse terms, and point-in-time evidence before adding any archive; do not substitute retrospective search dates for capture times.

For the Twitter archive, the next defensible experiment is a **separate** speaker-post-only and headline-plus-post feature ablation, restricted to event windows with frozen transcript references. Keep the headline-only baseline unchanged, deduplicate by permalink, exclude posts on/after the target date, disclose the source's timezone/deletion uncertainty, and report the smaller source-covered sample separately. Do not merge post text into the existing headline snapshot or claim that it restores the original news-only task. The UCSB source should remain out of automated ingestion pending permission or a user-provided export.

All post-source experiments remain retrospective development diagnostics. They do not establish independent or prospective accuracy, and they do not turn assistant-reviewed transcript inventories into objective semantic gold.
