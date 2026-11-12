# Backwards case study: Lexington rally, 2019-11-04

**Purpose:** start from one completed speech, inventory its transcript-derived ideas, then trace earlier news and speaker-account posts that mention related topics. This is a retrospective circulation study, not a prediction experiment, causal analysis, or semantic validation.

> **Correction / status:** the prior impeachment/Ukraine trace below is an exploratory keyword analysis with a construct error: it treats related material as one storyline. **Impeachment, the Ukraine call/complaint, and Biden-family Ukraine/China allegations are separate speech-level ideas and must not share a historical trajectory by default.** See [`2019-11-04-lexington-idea-breakdown-audit-v1.md`](../points/2019-11-04-lexington-idea-breakdown-audit-v1.md) for the provisional transcript-grounded breakdown. Do not use the combined counts below as evidence about the history of any one idea; rerun history only after the idea boundaries are audited.

## Speech and outcome inventory

Selected event: [Donald Trump rally in Lexington, Kentucky, 2019-11-04](https://rollcall.com/factbase/trump/transcript/donald-trump-speech-kag-rally-lexington-kentucky-november-4-2019/). It was selected because the local transcript is available and both a multi-outlet article archive and a historical Trump-account post archive cover the years preceding it. The rally occurred the day before Kentucky's 2019 gubernatorial election.

Codex (`openai-codex / gpt-6.1-sol`) extracted **63 transcript mentions linked to 54 candidate ideas**. All rows passed the deterministic quote/schema validator. **48 ideas** were marked forecastable by the model. This is specifically the inventory from Trump's target-speaker turns, not every speech on the rally stage: the event transcript also contains Rand Paul (163 words), Matt Bevin (94), Daniel Cameron (89), and Mitch McConnell (66). Their remarks remain separate speakers and were not included in Trump's 54 idea labels. The validator confirms quote structure, not whether an idea is a correct abstraction, complete, or semantically true. Do not treat the AI inventory as gold.

All 54 generated ideas, grouped for navigation:

- **Kentucky, candidates, and elections:** Kentucky Republican turnout; Bevin's performance; Beshear's political alignment; Beshear's economic agenda; McConnell endorsement; North Carolina election explanation; previously disengaged voters; campaign team credit; Rand Paul alliance; Kentucky pioneer legacy; 2020 Republican victory.
- **Impeachment, opposition, and media:** impeachment legitimacy; Republican unity/backlash; Russia investigation criticism; Democratic urban governance; Democratic economic threat; Biden competence/foreign dealings; media bias; media dependence on Trump; entrenched power; left-wing cultural coercion.
- **Economy, trade, and investment:** economic performance; manufacturing revival; tax cuts/opportunity zones; deregulation/energy; coal/climate rollback; China tariffs/negotiations; USMCA passage; TPP withdrawal; America First priorities; foreign defense payments; rural broadband.
- **Border, rights, and domestic policy:** sanctuary cities; border enforcement; immigration benefits/citizens-first; travel ban; Second Amendment; healthcare protections; opioid response; faith-based child services; late-term abortion restriction; constitutional freedoms; law-enforcement support; traditional civic values.
- **Security, foreign affairs, and science:** military strength; counterterrorism/ISIS; Iran nuclear deal; Jerusalem embassy; veterans' services; medical research goals; right to try; space exploration.
- **Event-local/ceremonial detail:** rally removal instruction.

The complete IDs, summaries, stances, prominence values, mentions, and transcript quotes are in the local outcome artifact named below.

## What was searched—and what “circulation” means here

The main media source is the locally pinned **All the News 2.1** article archive (27 listed publications, dates beginning 2016-01-01; archive revision `a1f7e19b50048259d6006324c8b12ee4a0284ac5`). The case-study query retained **2,326,017 distinct-URL article records dated 2016-01-01 through 2019-11-03**. We searched article **headlines**, not full text, retaining publication, author/byline, URL, and day. Exact duplicate URLs were collapsed; syndicated copies under different URLs remain and can inflate counts.

A second stream is the local Mark Hershey historical Trump Twitter archive, with **42,955 account-post records dated 2009-05-04 through 2019-11-03** in this pre-event slice. The archive has minute-formatted times without a timezone, no engagement counts, and no reliable original-versus-retweet field. Explicit `RT @handle` strings are counted only as visible repost text; absence of that prefix does not prove an original post.

Codex generated narrow search phrases from the 54 transcript-derived idea labels. I manually augmented a few Kentucky election phrases with candidate/state names so the search could retrieve the pre-election news vocabulary. Those phrase choices are retrieval aids, **not independent semantic annotations**. The case-study artifact records the terms matched on each example row. Headline phrase hits can be false positives (for example, markets coverage mentioning an impeachment inquiry) and can miss paraphrases. Counts below are therefore keyword-retrieval volumes, not audited counts of semantically relevant stories.

For each idea and stream, the script forms a daily series of matched items and total archived items that day. It reports matched share of daily corpus volume and applies an exposure-adjusted Poisson rate-state path over the **entire available history up to the event cutoff**. This does not impose a seven-day or other fixed lookback. Five rate states represent 1×, 2×, 4×, 8×, and 16× the full-series baseline rate; transitions pay a `log(number of days)` complexity penalty. The resulting “burst” is exploratory signal processing, **not** a statistical significance test; there is no multiple-idea correction, and results depend on phrase selection and archive coverage.

The event's start time could not be verified from the local record. Therefore **all event-day news and posts are excluded**; a same-day item is never assumed to predate the rally. Article dates have day-level precision. The news snapshot/archive was retrieved retrospectively, so published dates do not prove contemporaneous availability. The transcript is used only as the retrospective outcome and to create search terms, never as a same-event context feature.

## What the traces show

### Kentucky governor race: a clear event-linked lead-in

The transcript inventory contains central **Kentucky Republican turnout** and repeated **Bevin performance** ideas. On May 21, 2019, the archive has four matching headlines from four publishers and four bylines, including:

- [“Andy Beshear Wins Democratic Primary for Kentucky Governor”](https://www.nytimes.com/2019/05/21/us/politics/kentucky-primary-elections.html) — Jonathan Martin, *The New York Times*.
- [“Matt Bevin, Andy Beshear set for Kentucky clash”](https://www.politico.com/story/2019/05/21/beshear-to-face-bevin-in-kentucky-governors-race-1338125) — Daniel Strauss, *Politico*.
- [“Kentucky’s Democratic primary will pick a candidate to face Matt Bevin”](https://www.vox.com/policy-and-politics/2019/5/21/18623230/kentucky-democrat-governor-primary) — Catherine Kim, *Vox*.
- [“Trump-backed Kentucky governor narrowly survives primary”](https://thehill.com/homenews/state-watch/444933-trump-backed-kentucky-governor-narrowly-survives-primary) — Reid Wilson, *The Hill*.

The coverage returns at several distinct points: *The New York Times* reported on Oct. 29 that the governor’s race tested impeachment’s impact in states; *The Economist* framed the election as a referendum on Trump on Nov. 1; and *The Hill* published [“Trump makes last-minute pitch for Bevin campaign”](https://thehill.com/homenews/campaign/468732-trump-makes-last-minute-pitch-for-bevin-campaign) on Nov. 3 (Justine Coleman). The local Twitter archive also contains a Nov. 3 account post explicitly urging a vote for Bevin on Tuesday and saying “See you on Monday night”; the archived [post URL](https://twitter.com/realDonaldTrump/status/1191007379212648448) is retained in the local evidence row.

The phrase set finds 65 Kentucky-governor-related headline records over 48 active days and two account posts (one on May 21 and one on Nov. 3). The 65 are **not all election stories**: broad phrases such as “Kentucky governor” also match unrelated coverage of Bevin. The dated examples, rather than the aggregate, provide the strongest case evidence. This is a concrete pre-speech chain of political coverage → campaign post → rally outcome, but it does not show that the news caused the speech content.

### Impeachment: a large, sustained media burst plus account amplification

For the generated **impeachment legitimacy** idea, the neutral-topic search finds 703 headline records on 90 active days over the full pre-speech archive. The rate-state path identifies an elevated period from **Sep. 7 through Nov. 3, 2019**: 642 matching records across 53 active days, 18 publishers, and 246 bylines; maximum state 4 (the model's 16× baseline state). The peak date, Sep. 25, has 63 phrase-matched titles among 2,950 archive rows that day, from 13 publishers and 35 bylines (2.14% of that day's archive records).

#### Story-centred backtrace: Ukraine aid hold → call/complaint → impeachment inquiry

The transcript-derived impeachment idea is too broad to be the story entity, so this trace follows one specific news storyline that became central to the impeachment debate: the Trump administration's hold on Ukraine security aid, the July 25 Trump–Zelensky call, the whistleblower complaint, and the resulting House inquiry. The speech is used only to choose this retrospective outcome/story to trace; none of these same-event records are prediction inputs.

Separate **event occurrence** from **public discovery**. The July 25 call predates the speech by months. In the available All the News archive, the first clear report of the aid hold in this retrieved sequence is Politico's Aug. 28 [“Trump holds up Ukraine military aid meant to confront Russia”](https://www.politico.com/story/2019/08/28/trump-ukraine-military-aid-russia-1689531) (Caitlin Emma). On Aug. 29, CNN reported that Trump was considering blocking $250 million in aid, and Reuters reported officials said he might block it. The archive then shows a move from the aid question into scrutiny of the alleged pressure campaign:

| Date | Story/circulation milestone in the archive | Retrieval signal |
|---|---|---|
| Aug. 28–29 | Aid hold becomes publicly reported: Politico, CNN, Reuters; Pentagon's position covered by Politico. | These aid-specific articles are found by Ukraine+aid/Trump terms, not by the impeachment-only count below. |
| Sep. 3–6 | Bipartisan senators press for release; reporting continues on pressure to release military aid. [The Hill](https://thehill.com/homenews/senate/459838-bipartisan-senators-urge-trump-administration-to-release-ukraine-aid), [Politico](https://www.politico.com/story/2019/09/03/trump-ukraine-russia-bipartisan-letter-1479880), [Politico, Sep. 6](https://www.politico.com/story/2019/09/06/ukraine-military-aid-1483770). | Several distinct publishers and reporters cover the aid hold and its policy rationale. |
| Sep. 9–12 | House committees open a probe into Trump, Giuliani, and Ukraine; the administration releases the $250 million after bipartisan pressure. [Reuters](https://www.reuters.com/article/us-usa-trump-ukraine-giuliani/house-committees-launch-probe-of-trump-giuliani-and-ukraine-idUSKCN1VU1SA), [The New York Times](https://www.nytimes.com/2019/09/09/us/politics/trump-ukraine-house-investigation.html), [Politico on release](https://www.politico.com/story/2019/09/12/trump-administration-ukraine-military-aid-3825755). | On Sep. 12 alone the broad title query has 23 records, 10 publishers, and 18 byline strings; not all 23 are aid-hold stories. |
| Sep. 19–21 | Reporting identifies Ukraine in the whistleblower complaint; reports say Trump pressed Zelensky to investigate Biden; Biden and Trump respond publicly. [The New York Times, Sep. 19](https://www.nytimes.com/2019/09/19/us/politics/intelligence-whistle-blower-complaint-trump.html), [Reuters, Sep. 20](https://www.reuters.com/article/us-usa-trump-whistleblower/trump-pressured-ukraine-president-to-investigate-biden-reports-idUSKBN1W51R0), [The New York Times, Sep. 20](https://www.nytimes.com/2019/09/20/us/politics/trump-whistle-blower-ukraine.html). | The story expands from a hidden aid decision into the call, the complaint, Biden, Giuliani, and competing accounts. |
| Sep. 24–26 | Pelosi announces an impeachment inquiry; the call record is released Sep. 25 and the whistleblower complaint becomes public Sep. 26. | The broad title matcher returns 201 records/15 publishers/103 byline strings Sep. 24; 258/18/151 Sep. 25; and 226/18/139 Sep. 26. Sep. 25 examples include [The New York Times on Democrats' reaction](https://www.nytimes.com/2019/09/25/us/politics/2020-democrats-trump-impeachment.html) and [BuzzFeed News on moments in the call](https://www.buzzfeednews.com/article/zoetillman/trump-ukraine-call-transcript-biden-impeachment). |
| Oct. 3–17 | A second whistleblower story and then public testimony/deposition developments sustain coverage; Oct. 17 coverage includes Ambassador Gordon Sondland arriving to testify. | The same broad matcher returns 66 records/13 publishers/46 byline strings on Oct. 3 and 29/10/23 on Oct. 17. These are candidate volume counts, not adjudicated relevant-article counts. |
| Oct. 31–Nov. 3 | The House votes to formalize inquiry procedures; by Nov. 3, reporting turns to the next stage and an attorney's offer to submit questions to the whistleblower. [The New York Times, Oct. 31](https://www.nytimes.com/2019/10/31/us/politics/house-impeachment-vote.html), [Politico, Nov. 3](https://www.politico.com/news/2019/11/03/republicans-questions-trump-whistleblower-064767). | Oct. 31 has 108 records/12 publishers/77 byline strings; Nov. 3 has 43/10/24. |

These figures come from a broad headline retrieval query (`impeach`, `whistleblower`, `Ukraine call`, or `Ukraine memo`) across Aug. 28–Nov. 3: 3,052 distinct-URL records from 23 publishers and 719 non-null byline strings, with the first returned record dated Aug. 29. It is **not** an audited corpus of the Ukraine story: the query includes generic impeachment and unrelated whistleblower headlines, while the initial Aug. 28 aid story required separate Ukraine-aid terms. The article-date sequence supports a circulation reconstruction, but not proof that any item was available to a particular person in real time.

The local Trump-account archive adds a different layer. A broad account-text query for Ukraine, impeachment, whistleblower, or Zelensky returns **211 posts on 40 active days from Sep. 19 through Nov. 3**, including **102 posts with visible `RT @handle` text**. Examples include the Sep. 25 post announcing release of the call record, Sep. 26 visible reposts from political figures/media accounts, and Nov. 3 posts about the whistleblower. Counts peak at 14 on Sep. 26, 13 on Sep. 29, and 11 on Oct. 29. This is Trump-account posting/reposting, not a measure of audience diffusion; the query includes broad Biden/Ukraine and impeachment material and the archive has no engagement counts or timezone. The previously reported narrower impeachment-phrase result (34 posts on 16 days) remains a separate query, not a contradiction.

The titles show which actors were being discussed: Pelosi's announcement, Schumer's response, Senate Democrats, Adam Schiff, Ukraine, and the inquiry process. Examples from that date include Axios headlines by Rashaan Ayesh on Schiff's comments and Pelosi's response, and Business Insider coverage by John Haltiwanger on House support for an inquiry. The most frequent bylines during the detected period were Noah Weiland (25), Zachary B. Wolf (15), Zack Budryk (12), Justine Coleman (11), and Nicholas Fandos (10); the most frequent publishers were *The Hill* (160), CNN (110), *The New York Times* (91), Reuters (76), and Vox (41). Bylines are article authors, not necessarily the people quoted.

The account archive has 34 phrase-matched posts on 16 days from Sep. 25 through Nov. 3; the detector flags an elevated state over that interval. Its peak date, Oct. 25, has three matching posts among 14 archived account posts that day. Across the interval, explicit `RT @...` text most often names `RepLeeZeldin` (4), `VP` (2), and `Jim_Jordan` (2), followed by `RepDougCollins` (1), among other handles. An Oct. 25 post thanks Lindsey Graham and Mitch McConnell for a resolution condemning the inquiry, alongside a visible retweet from Rep. Lee Zeldin. This is evidence of account-posting/reposting around the issue, **not** evidence about likes, reach, or downstream user virality.

The broad phrase matcher also catches non-political market headlines that mention impeachment. Treat the total and burst as a candidate circulation signal; a next-stage evidence audit should classify each headline's topical relevance before interpreting those rates as political-discourse volume.

### Other long-running ideas

- **China tariffs/negotiations:** 989 matching headlines over 291 active days; the detected elevated run is Feb. 11–Nov. 3, 2019, with 714 matches on 165 active days, 12 publishers, and 167 bylines. The peak is Mar. 4 (22 matches; 1.31% of that day's archive). Top publishers in the run include CNBC and Reuters. This looks more like a sustained story stream than a seven-day spike.
- **Border enforcement:** 1,954 matches over 589 active days. The strongest detected run is Nov. 22, 2018–Apr. 9, 2019 (803 matches, 123 active days, 21 publishers, 307 bylines); Dec. 21, 2018 peaks at 31 matches (2.13% of that day's archive). Trump-account archive text also contains repeated border-related posts. These broad counts need precision review because the search includes general border-security language.
- **McConnell/McGrath:** 586 matching news headlines; an elevated period May 19–Aug. 9, 2019 includes 111 records from 19 publishers and 77 bylines. That campaign signal precedes the rally by months, although some hits about McConnell concern other subjects and are not necessarily the reelection storyline.

These results expose the pattern the seven-day slice cannot express: some ideas are **event-specific late bursts** (the Kentucky race and impeachment), while others are **persistent or recurrent background streams** (border enforcement, trade, and McConnell). A model should represent an issue's trajectory, burst onset, persistence, outlet diversity, and speaker-post activity—not only a count inside one arbitrary fixed window.

## Limits and next step

This is a useful first backwards trace, not a virality study. The local sources provide headline publication volume, outlets, bylines, and one speaker-account archive. They provide no reliable audience impressions, historical likes/shares, reply networks, or general-public posting volume for 2019. Article counts are not reach; an archived post is not a viral post. The source set is incomplete, mostly national, and excludes local Kentucky outlets that may be especially important here. Post retweet/original status and timezone are imperfect, and current archive presence does not prove historical visibility.

Next, use the saved daily records to build a **topic-relevance audit** for candidate article/post matches, then add a source with historical engagement or network data before calling any signal “virality.” Only after point-in-time source availability is established should a separate, chronological context-to-idea model be trained. Do not move transcript-derived labels into the event's same-event input.

## Reproducibility and local-only artifacts

Research code:

- `research/idea_layers/stories/speech_news_circulation_case_study.py` — source joins, deduplication, daily exposure series, and burst-state detector (`uv run --with duckdb ...`).
- `research/idea_layers/stories/plan_speech_idea_search_terms.py` — Codex phrase-plan generator. Case-specific manual term additions are recorded in the term-plan JSON.

Ignored local artifacts (no transcript, post, article, or generated record is committed):

| Artifact | SHA-256 |
|---|---|
| `2019-11-04-lexington-rally-outcomes-codex-v1.jsonl` | `c61de0cc0564540a0ad6cc35a434aad6b484b03a622b6dc1b24cb3354d077a4e` |
| `2019-11-04-lexington-search-terms-case-curated-v2.json` | `5999cd6ae4dd59decfbde5317b1a110eb42cef85928e9cb710e9086d41438591` |
| `2019-11-04-lexington-news-circulation-v8.jsonl` | `8e82fc1d4ff080341117214dd08718d9afc207f732a0ff5e6764295d71941f8f` |
| `2019-11-04-lexington-news-circulation-v8.jsonl.manifest.json` | `e30ed0bce607ba2fc57b55611757488212b5c9d0283fd069382c89b6f7f65ebc` |

The output manifest pins the event corpus, every Parquet/Twitter input file, term plan, timestamps, and limitations. Codex model aliases are mutable. The licensed-use status of the underlying All the News article text is unclear; only local analysis was performed, and the raw corpus and derived evidence must not be redistributed.
