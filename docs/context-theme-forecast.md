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

A prospective v2 record must include timezone-aware RFC 3339 values for event start and prediction cutoff; event-metadata provenance independently verified as available before the cutoff; exact retrieval/observation timestamps for queries and candidate sources; and publication timestamps strictly before cutoff. Use original publisher URLs and timestamps where possible, not a date-only search-feed date. Location may be included only with independent evidence it was known before cutoff. `speechprefix context-snapshot-seal --input <prepared.json> --output data/rollcall-context-prospective-<event-id>.json` stamps the current clock time, requires the cutoff and event to be in the future, validates provenance fields, and adds a content hash; `context-snapshot-validate` verifies the recorded contract and hash. The seal is local and append-only but cannot independently prove that source timestamps or availability attestations are truthful; retain and review the source documents and capture procedure. It also rejects transcript/target-label fields. Never relabel a retrospective v1 snapshot as prospective.

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

`speechprefix transcript-idea-audit` mines target-speaker, named-entity-anchored phrases from transcripts (with cited turn IDs) and checks whether all phrase terms appear together in a dated context headline. It has no assistant labels or human annotations. The extractor can run on any observed-turn prefix as well; the current report uses full transcripts solely to derive post-hoc idea outcomes, not as prediction input.

The 2024-10-21 Henderson transcript yields the idea **“war Ukraine”** in Trump turns `dt45-24-883c1eff:turn:0056` and `:turn:0213` (“I will end the war in Ukraine”). None of the 12 retrieved pre-event-dated headlines contains both terms. This is the direct failure: the transcript idea is easy to identify automatically, but the context snapshot did not feed the model evidence for it. The snapshot was retrieved retrospectively, so this is a diagnostic, not a prospective forecast score.

Across 20 events, top-ranked headline phrases were literally echoed in the transcript at 2.5% precision@10 (5/200 phrases); only 7/200 top-25 phrases and 18/1,000 top-50 phrases matched. Conversely, only 7/200 top-10 transcript-derived target phrases had a matching headline (3.5% coverage). These low lexical-overlap numbers reflect noisy context retrieval and a deliberately literal matching rule; they are not semantic accuracy or probabilities. The full citations and phrases are in the ignored artifact `data/rollcall-transcript-idea-audit-v2.json`.

## Next gates

1. Improve and relevance-audit retrieval without consulting transcripts; verify event metadata and location were available before each event.
2. Add chronological pairs with both observed and absent candidate ideas.
3. Keep context-derived issue ideas distinct from broad families and evaluate those levels separately.
4. Compare against metadata-only and speaker-history baselines; calibrate only after a substantially larger independent holdout.
5. Review errors before considering more complex models or expanding beyond themes.
