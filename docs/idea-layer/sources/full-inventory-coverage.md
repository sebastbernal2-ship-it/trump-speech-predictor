# Source-date overlap across the transcript inventory

Assessed 2026-11-17 against the supplied local snapshots, using [inventory_coverage.py](../../../research/idea_layers/sources/inventory_coverage.py). This is a **coverage screen, not verified pre-cutoff coverage or permission to use content**. It does not select future inputs. The four inventory views contain overlapping records, not 9,642 distinct speeches; eligibility, event identity, source versions and event dates remain unverified. The Roll Call view includes records as early as 1980, so the earlier 1988–2026 description is not the range of this full inventory.

For each row, the script tests whether at least one source record has a stated date on any of the seven calendar dates preceding the event date. It excludes the event date. It counts rows with a hit, not articles, posts, relevant content, publishers or complete coverage; no deduplication is required for this binary test. It uses the UTC date in Truth Social's `created_at` and the unverified date/time zone of Twitter's `Time`, without converting time zones. Even a prior calendar date does not independently prove actual availability before a speech, especially with unknown event cutoff and time zone. Seven days is a common screening window, not a proposed optimal horizon. All 9,642 rows had parseable dates under this test.

| Transcript view | Rows | ABC | HuffPost | Twitter | Truth Social | Either social |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Roll Call events | 5,483 | 3,081 | 1,343 | 3,076 | 2,314 | 5,390 |
| Events | 1,940 | 0 | 0 | 0 | 1,940 | 1,940 |
| Speeches | 1,019 | 525 | 520 | 525 | 494 | 1,019 |
| Long speeches | 1,200 | 741 | 394 | 742 | 436 | 1,178 |

HuffPost hits are a subset of ABC hits in each view. For the Roll Call view, neither of these two news snapshots has a hit for any of its 2,329 rows dated 2022–2026; the social snapshots have no hit for 31 of 41 rows in 2021 and 15 of 61 in 2022. These are gaps in these snapshots, not proof that no news or posts existed. For year-level counts and the other views, rerun the script: `python3 research/idea_layers/sources/inventory_coverage.py /path/to/data`. It reads local files only and emits aggregate JSON lines. Its inputs are the four inventory files and the four raw archive files described in the [survey](historical-data-survey.md), including both Twitter CSVs. The all-the-news and CC-News Parquet snapshots, other sources and the benchmark's separate 38-event windows were **not scanned** here; their absence from this table is not zero coverage. The prior audit's source manifests record snapshot identities and hashes; refresh and hash all inputs when reproducing on changed data.

## Rights and cutoff evidence

| Retained source | Rights evidence examined | Timing evidence and disposition |
| --- | --- | --- |
| ABC | Dataset declares CC0; publisher rights to headlines remain unverified. | Day only, no URL; retrospective copy provides no independent first-availability or version evidence. Hold for rights and point-in-time review. |
| HuffPost | Dataset mirrors report conflicting CC BY 4.0 and CC0 claims; publisher rights remain unverified. | Day only; retrospective copy does not establish first availability or revisions. Hold. |
| Twitter archive | Repository MIT license does not establish rights to posts. | Time zone and historical deletion/visibility unverified. Hold. |
| Truth Social archive | Archive CC0 claim does not establish rights to posts or media. | UTC creation timestamp is not proof of visibility at cutoff; deletion and revision history absent. Hold. |
| All the News / CC-News | Prior audit found no clear reuse license; underlying rights are unresolved. | Snapshot dates are not independent availability evidence; full-inventory overlap is unmeasured. Hold. |

These determinations rely on retained source documentation and the [earlier audit](input-source-audit.md), not a legal opinion or confirmation from rights holders. No archive has established both applicable reuse rights and per-item cutoff-time availability in this assessment. Unknown is not a negative finding that an item was unavailable, and a timestamp alone is not a positive verification. In particular, the inventory's date is not an independently established speech start time.

Before considering a source as an input, record the applicable permission and permitted uses from the rights holder or governing terms, preserve its evidence and date, and establish an event-specific cutoff with time zone. For each candidate item, retain an immutable record and provenance, capture or other independent evidence that the specific version was available by that cutoff, and mark uncertain revisions, deletions and first availability unverified. Quantify per-event coverage, missing timing, and modality separately; avoid same-event transcript leakage. Resolve the unscanned archives and older and recent news gaps before treating this screen as a source selection. Any eventual comparison requires a separately versioned chronological evaluation against the unchanged frozen baseline.
