# Context-to-theme forecast baseline

## Task

This experiment ranks broad theme families that may appear in a future speech from dated context headlines. It is separate from transcript continuation, question-response prediction, and explicit plan-path generation. The current fixed families are elections/governance, economy/jobs, immigration, national security, public safety, health, culture/education/religion, media, energy/environment, and disasters/local resources. A separate lexical candidate layer can surface source-grounded topics such as a named controversy, but does not generate arbitrary semantic ideas or establish their likely inclusion in a speech.

The CLI is:

```bash
speechprefix context-theme-evaluate \
  --snapshots data/rollcall-context-snapshots-v1.jsonl \
  --labels data/rollcall-theme-family-drafts-v1.jsonl \
  --output data/rollcall-context-theme-evaluation-v1.json
```

The evaluator fits one Bernoulli naive-Bayes classifier per family on the earliest 80% of event dates, reserves the next 10% for validation, and evaluates the final 10% chronologically. Inputs are source-headline tokens only; event titles, transcript text, and target theme summaries are excluded. It compares with a training-prevalence baseline and records data hashes, split IDs, predictions, evidence headlines, and metrics. Reports are immutable and the command refuses to overwrite an existing artifact.

## Pilot data and policies

`data/rollcall-context-snapshots-v1.jsonl` contains 20 metadata-only retrieval snapshots and 234 dated headline candidates. Queries used a 30-day pre-event window; same-day and later sources were excluded. Headline candidates are not relevance-reviewed. Event locations were inferred from retrospective titles, and their availability before each speech is unverified. The collection is exploratory, not a strict prospective benchmark.

`data/rollcall-theme-family-drafts-v1.jsonl` contains assistant-authored broad-family mappings from the full-transcript theme drafts. The source theme drafts and family mappings are unreviewed. Omitted families are weak inferred negatives, not verified transcript absences. All source records, draft labels, context maps, and evaluation reports remain local and Git-ignored.

## Prospective snapshot gate

The v1 snapshot set is not a strict pre-event benchmark. A headline's date being earlier than the speech does not show that the headline was retrieved before the speech, and retrospective location-based search can select stories with hindsight. `speechprefix context-snapshot-audit` records these deficiencies; the current 20-snapshot audit found **0/20** eligible for prospective evaluation. All 20 were retrieved on/after their event date, lack exact event-start and cutoff timestamps, and lack per-source observation timestamps. All candidate publication dates are day-precision only; 17/20 use location hints whose pre-event availability is unverified.

A prospective v2 record must include timezone-aware RFC 3339 values for event start and prediction cutoff; event-metadata provenance independently verified as available before the cutoff; exact retrieval/observation timestamps for queries and candidate sources; and publication timestamps strictly before cutoff. Use original publisher URLs and timestamps where possible, not a date-only search-feed date. Location may be included only with independent evidence it was known before cutoff. `speechprefix context-snapshot-seal --input <prepared.json> --output data/rollcall-context-prospective-<event-id>.json` stamps the current clock time, requires the cutoff and event to be in the future, validates provenance fields, and adds a content hash; `context-snapshot-validate` verifies the recorded contract and hash. For fixed speaker/topic-family Google News queries, `speechprefix context-news-freeze --manifest <event.json> --output data/rollcall-context-prospective-<event-id>.json` performs the retrieval and seals it in one operation. The manifest must provide `event_id`, `event_start_at`, `prediction_cutoff_at`, and `event_metadata` with `source_url`, `published_at`, `observed_at`, and `availability_verified_before_cutoff: true`; all times are timezone-aware. For example:

```json
{
  "event_id": "scheduled-event-id",
  "speaker": "Donald Trump",
  "event_start_at": "2030-01-20T18:00:00Z",
  "prediction_cutoff_at": "2030-01-20T16:00:00Z",
  "event_metadata": {
    "source_url": "https://example.org/official-schedule",
    "published_at": "2030-01-19T12:00:00Z",
    "observed_at": "2030-01-19T12:05:00Z",
    "availability_verified_before_cutoff": true
  }
}
```

It refuses to start after cutoff, excludes articles published after observation or at/after the exact cutoff, records query/source observation times and response hashes, and fails if retrieval finishes too late. The fixed query profile does not use event titles, transcripts, or target ideas. Google RSS timestamps remain feed-reported timestamps, not independently verified publisher times. The seal is local and append-only but cannot independently prove that source timestamps or availability attestations are truthful; retain and review the source documents and capture procedure. It also rejects transcript/target-label fields. Never relabel a retrospective v1 snapshot as prospective.

Before the cutoff, `speechprefix context-issue-forecast --snapshot <sealed.json> --output data/rollcall-context-issue-forecast-<event-id>.json` writes a hashed, immutable list of compact issue phrases with source citations. It validates the sealed snapshot and refuses to run at or after cutoff. These are context-supported candidates, not probabilities or claims about what Trump will say; the retrospective chronological test found this context-only ranking worse than the speaker-history baseline.

## Confidence policy

The Bernoulli naive-Bayes posterior is emitted as an **uncalibrated ranking score**. It must not be presented as a reliable probability or confidence estimate until calibration and reliability are demonstrated on a substantially larger, independent chronological holdout. A saturated score is not evidence of certainty by itself; judge it by accuracy and calibration on future held-out cases.

## Initial results and limits

The v1 split is 16 train / 2 validation / 2 test events. On the two-event test, context Naive Bayes and the training-prevalence baseline both had micro-F1 **0.727**, macro-F1 **0.433**, mean precision@4 **0.875**, and mean recall@4 **0.75**. The context model's uncalibrated Brier score was **0.299**, worse than the baseline's **0.164**. It did not improve top-four selection on this tiny holdout; its posterior values saturated near 0 and 1 and its evidence included generic, irrelevant terms. These are weak-label diagnostics, not semantic-accuracy claims. Two events cannot establish generalization or calibration.

The frozen Qatar and Henderson context maps include assistant-authored theme drafts and turn-level outcomes. Their manually curated candidate lists are not independent benchmarks and contain no negative candidates; their 4/4 broad-topic overlap is not model performance or predictive lift. The lexical candidate generator finds the four intended source topics in Henderson and four related topics in Qatar, but these same curated sources make this only a functionality demonstration.

In an exploratory replay of Henderson's frozen context, the v1 ranker (trained on the earlier 16 pilot events) ranked economy #1, election #2, foreign policy #3, immigration #4, and media #9. It missed the desired garbage-controversy/media idea from its top four and did not name specific Nevada-turnout or working-class themes. This is a concrete failure of the requested specific idea selection despite the broad-family overlap. The detailed replay and hashes are in the ignored local artifact `data/rollcall-context-theme-case-replay-v1.json`.

## Source-derived candidate diagnostic

`speechprefix context-candidates --input <context-map-or-snapshot-jsonl> [--event-id ID]` extracts lexically matched topic candidates with dated source citations. This is a candidate-generation aid, not a probability model. `speechprefix context-candidate-evaluate` compares its fixed lexical candidates, ranked by source count/diversity/recency, with the training-prevalence baseline.

On the same exploratory 16/2/2 split, source-evidence top-four achieved mean precision@4 **0.50** and mean recall@4 **0.417**, below training-prevalence top-four (**0.875** and **0.75**) on two test events. The unranked source candidate set had mean precision **0.833** but mean recall **0.417**, indicating that the lexical layer can retrieve plausible candidates while missing many weakly labeled families; its ranking did not beat the trivial baseline. All results use unreviewed sources and assistant-drafted labels and are not semantic accuracy estimates. A new immutable diagnostic is saved locally as `data/rollcall-context-candidate-evaluation-v2.json`; do not overwrite it. The v1 report is retained as a prior artifact from before tightening diplomacy/Israel matching.

## Human annotation pilot

`speechprefix theme-review-packet` creates two identical, independently fillable transcript-only copies for the 20-event pilot plus a separate event-ID key. The annotator forms omit event IDs, titles, dates, URLs, assistant draft labels, model predictions, and news context. Reviewers label every broad family present/absent/uncertain with turn citations, then enter up to five event-specific ideas with supporting turns and stance. Keep the key separate from the annotators and have each reviewer work independently. The local files are `data/rollcall-theme-review-annotator-{a,b}-v1.jsonl` and `data/rollcall-theme-review-key-v1.json` (all Git-ignored).

These forms are not annotations or gold labels. This small, context-selected event sample is only suitable for testing instructions, schema, and inter-annotator agreement. After independent completion, `speechprefix theme-review-agreement` validates cited turns and reports family-level raw agreement, Cohen's kappa, positive/negative agreement, and disagreements requiring adjudication. It does not automatically adjudicate free-text ideas or convert drafts to gold. A human adjudicator must create one decision JSONL row per `review_item_id`, enter every family's final `present`/`absent`/`uncertain` label and evidence turns, and resolve disagreements with a note. `speechprefix theme-review-adjudication-template` creates the blank adjudication form; after humans complete it, `speechprefix theme-review-adjudicate --decisions <human-decisions.jsonl>` validates the decisions and exports source-hashed human-adjudicated labels. Blank forms cannot be exported. Uncertain families remain explicit rather than becoming negatives. The taxonomy also needs review; disagreement should be adjudicated before model evaluation. The candidate list is intentionally not shown to prevent anchoring; event-specific candidate recall will require a separate post-annotation adjudication step.

## Transcript-derived idea check

`speechprefix transcript-idea-audit` now extracts compact issue phrases rather than arbitrary adjacent words. It strips bracketed stage cues, requires a topic anchor or an allowed geopolitical/concept pair, and excludes unanchored person-name fragments and arbitrary anchor combinations. Every result retains turn citations. The headline side uses the same phrase gates. This is a lightweight rule-based filter, not a complete ontology or semantic model; phrases outside its vocabulary can be missed. It uses no assistant labels or human annotations. The 20-event audit uses full transcripts only to derive post-hoc outcomes, never as pre-speech input.

The pass covers all 20 transcript/context pairs. Examples include **Eminent domain**, **Keystone pipeline**, **trade deficit**, and **Second Amendment** (2016-01-24); **North Korea**, **South Korea**, and **European Union** (2018-03-10); and **Border Patrol**, **energy costs**, and **Ukraine war** (2024-10-21). The last event's Ukraine-war mentions are cited in turns `dt45-24-883c1eff:turn:0056` and `:turn:0213`.

In the latest rule-filtered report, only 12 headline-derived multiword candidates survived across 20 events, and 1 was echoed in its transcript (8.3%; too small a denominator for a useful precision estimate). The top-10 transcript phrase inventory had 1 headline match among 167 phrases (0.6%); across top 50, 1/243 (0.4%). These are literal phrase-match diagnostics under the current filter—not semantic accuracy or prospective performance. The restrictive lexicon improves readability at the cost of unknown recall. Context snapshots are retrospective, and publication times are day-level. Full cited output is in the ignored artifact `data/rollcall-transcript-idea-audit-v17.json`.

## Speaker-name retrieval diagnostic

`speechprefix context-news-search` provides metadata-only retrieval profiles with date-bounded Google News RSS queries; neither profile uses transcript text or extracted target ideas. The speaker-name profile uses only the speaker and event dates. Its 20-event, 30-day run retrieved 987 candidates to `data/rollcall-context-snapshots-trump-name-v2.jsonl`; query strings, retrieval times, source dates, and response hashes are retained.

Compared with the original location-plus-generic-national-news snapshots, the speaker-name query raised literal matching of ranked headline phrases from 1/12 to 7/29 and top-10 transcript-phrase coverage from 1/167 (0.6%) to 7/167 (4.2%). This remains an exploratory retrieval diagnostic, not a prospective forecast: searches ran after the events against today's index. One counted “American workers” match came from the UAW organization's name, illustrating incidental lexical matches. The search also found Tax Foundation's “Trump’s Tax and Tariff Ideas: Details & Analysis” (2024-10-14), a broad tax/tariff signal but not the transcript's specific “tax overtime” idea. The latest report is `data/rollcall-transcript-idea-audit-trump-name-v5.json` (ignored locally).

The `speaker-topic-families` profile adds four fixed, broad searches (economy/trade, immigration/border, foreign policy, and domestic policy) to the speaker-only query; the same queries run for every event. Across 20 events it retrieved 4,265 unique candidates. With the same phrase filter, top-10 transcript-phrase coverage rose to **37/167 (22.2%)**; **43/170 (25.3%)** ranked headline phrases were echoed in the transcripts. This gives the context a much richer issue signal, but it also returns generic and incidental matches, and the higher coverage comes from a much larger candidate pool. For the 2024-10-21 speech, the query found 11 headlines supporting “Ukraine war,” including Reuters' 2024-09-28 “Zelenskiy says Trump assured him of support for Ukraine in war with Russia.” That's relevant pre-event-dated evidence, but retrieval happened after the speech and does not prove what was available at its cutoff. Treat all these figures as retrospective lexical diagnostics, not forecast performance. Snapshot and report: `data/rollcall-context-snapshots-topic-families-v1.jsonl` and `data/rollcall-transcript-idea-audit-topic-families-v3.json` (ignored locally).

## Idea-level chronological test

`speechprefix context-idea-evaluate --snapshots <snapshot.jsonl>` compares the context phrase ranker with a speaker-history baseline: the most frequent automatically mined idea phrases from earlier transcripts. The chronological split is 16 train / 2 validation / 2 test; the test events are both 2026 rallies. On the two-event holdout, context-only top-four scored precision **0.125**, recall **0.063**; history prevalence scored **0.625** and **0.299**. At top ten, context scored **0.10 / 0.118**, history **0.45 / 0.528**. So the larger retrieval pool surfaces many true ideas somewhere in its candidates, but the current evidence ranking does not select them better than recurring speaker-history themes. The test is far too small for a general claim, and both target phrases and retrieval are exploratory/retrospective. The immutable local comparison is `data/rollcall-context-idea-evaluation-topic-families-v2.json`.

## Next gates

1. Use `context-news-freeze` for a genuinely future scheduled event and independently verify the schedule-source availability before cutoff; no such snapshot is collected by the retrospective experiments above.
2. Improve and relevance-audit retrieval without consulting transcripts; verify event metadata and location were available before each event.
3. Add chronological pairs with both observed and absent candidate ideas.
3. Keep context-derived issue ideas distinct from broad families and evaluate those levels separately.
4. Compare against metadata-only and speaker-history baselines; calibrate only after a substantially larger independent holdout.
5. Review errors before considering more complex models or expanding beyond themes.
