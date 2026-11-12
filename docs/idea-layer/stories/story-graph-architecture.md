# Story-level news context and graph-ready architecture

**Status:** experimental input-representation design. The active seven-day headline classifier remains the frozen comparison baseline. `speechprefix/idea_layer/story_graph.py` currently provides versioned graph validation, provenance checks, and as-of projection only; it does **not** cluster news, score relevance, build entity profiles, or implement a GNN.

## Why this layer comes before a new predictor

A feed row is not an independent signal. It may be a syndicated copy, a rewrite, a correction, or a genuine new development in an ongoing situation. Counting rows or retaining the newest fixed number can overweight distribution and repetition while discarding an older story that remains active. Therefore preserve source-level observations and their relations first; derive story-level features second; only then evaluate how those features should influence an idea forecast.

Do not collapse every related headline into one irreversible cluster. Preserve article observations and represent relations separately:

- **Exact duplicate / syndication:** same canonical URL with unchanged content, or strongly supported exact-content matches across URLs. Keep every publisher observation and attach an identity relation so source breadth can be measured without counting copies as independent corroboration.
- **Update / correction / follow-up:** a later version at the same URL with changed content, or a later observation with new facts, correction, or material change. Keep it as a distinct time-ordered development, not another independent vote and not a duplicate.
- **Related coverage:** topical or entity overlap that does not establish the same underlying story. Keep as a weaker typed relation; never merge solely because two stories mention the same prominent person or issue.

Cluster confidence and relation evidence must be retained. Uncertain links should remain candidate links rather than forced merges. Cluster IDs and relation creation must be reproducible and point-in-time safe; a future article must not retroactively change the story representation available to an earlier target.

## Graph contract

The graph is heterogeneous and queryable by time. Version 1 supports these node types:

- `article_observation`: one source item/edition, preserving source identity and title/content hash.
- `publisher`: a publishing outlet, linked to observations so outlet breadth and ownership/syndication dependence can be distinguished.
- `story`: an evolving situation, with observation memberships and a development timeline.
- `entity`: a canonical person, organization, place, or other actor/concept.
- `entity_claim`: one dated profile fact or relationship claim, not a timeless biography blob.
- `concept`: a source-derived issue/topic node.
- `target_event`: the forecast query context, without its transcript or outcome labels.
- `idea_candidate`: a candidate query node; never a realized transcript label for the target event.

Typed edges include `member_of`, `published_by`, `owned_by`, `exact_duplicate_of`, `syndicated_copy_of`, `updates`, `related_to`, `mentions`, `claim_about`, `associated_with`, `employed_by`, `met_with`, `endorses`, `opposes`, `located_in`, and `candidate_evidence`. Add relation types only with evidence and a versioned schema change. Each node and edge carries source IDs, availability status, temporal metadata, and (for edges) an explicit confidence score. Graph metadata records the algorithm version and whether it is deterministic or learned; learned algorithms must record their training cutoff, which strict as-of projection checks against the event cutoff. An edge's `available_at` cannot precede either endpoint's availability. Source IDs resolve to the immutable evidence ledger; the graph contract validates their presence when that ledger is supplied. As-of projections seal both the graph and a hash of the evidence records actually cited in that projection, excluding evidence belonging only to future nodes.

`story_graph_as_of(...)` makes a strict subgraph using only records and cited sources verified available before the cutoff. Its explicitly labeled `retrospective` mode may use pre-cutoff publication timestamps, but date-only items on the cutoff date are excluded and the result is marked `point_in_time_verified: false`. This is a research convenience, not proof that an item was historically visible. Profiles and relations follow the same rule: a present-day biography cannot be projected backward without dated, pre-cutoff evidence.

## Story reduction and learned relevance

The first corpus pass should report a reduction ledger for each source collection and event: raw observations, exact URL/content duplicates, high-confidence syndicated copies, candidate update/related links, independent publishers, and resulting story/development counts. Every reduction is reversible and records algorithm version, evidence, and confidence. Because no human annotation is planned, quality claims must stay limited to reproducible checks (exact-identity recovery, synthetic adversarial fixtures, temporal integrity, and sensitivity analyses); do not call unreviewed semantic clusters ground truth.

### Initial 2019 exact-identity probe

`research/idea_layers/stories/2019_story_reduction_audit.py` hashes normalized article bodies locally and emits aggregate counts only; article bodies are not graph/model features in this probe. Across 655,454 All the News 2.1 2019 rows with nonempty titles, 616,116 have nonempty bodies and 570,645 distinct normalized body hashes. There are **39,990 repeated-body-hash groups** (45,471 excess rows); **579 groups** span multiple URLs and publisher values. Only two exact URL groups repeat, with no repeated-URL group showing a changed title/body hash. This is evidence of exact-text redundancy, not proof that every repeated body is a syndicated story; boilerplate and low-information pages need safeguards.

The same corpus has 57,359 repeated normalized-title groups, 17,436 spanning multiple publisher values (16,937 same-day). ABC has 34,060 2019 headlines and 868 excess rows with repeated normalized titles. Cross-corpus matching found 19 same-day exact-title pairs (11 title/date combinations). Title matches are only candidate links: identical titles can describe distinct events, and differing titles can report the same event. The repeat-URL result also indicates that update detection cannot rely on stable URLs alone.

The full report is local and Git-ignored at `data/source-archives/2019-story-reduction-audit-v2/2019-story-reduction-audit-v2.json` (SHA-256 `37fd56ddc04552f3ecf10695a8bf8915b4b1ecdab8af992a397aaaf11600eb99`). It contains counts and source-manifest hashes, not article text. Reproduce with `uv run --with duckdb python research/idea_layers/stories/2019_story_reduction_audit.py` after placing the pinned source archives locally.

Do not install a universal hand-picked source weight, article cap, seven-/fourteen-day horizon, or recency half-life as though justified. The older `idea_layer/ranker.py` has illustrative fixed source weights and a 14-day half-life; those are explicitly not the story-graph defaults. Preserve continuous time signals—age of first report, age of latest substantive update, time since correction, update cadence, and source-observation time—and features for independent-source breadth, duplicate volume, novelty, and entity/query relevance. Keep raw counts and normalized/cluster-level versions for ablation. The archive boundary is a data-availability limit, not a claim that older context is irrelevant.

For an initial comparison, retain the existing seven-day item-level baseline, then evaluate story-level inputs over the broader pre-cutoff archive. Learn any time decay, attention, or source aggregation only on chronological training/validation folds. Use whole-date grouping, embargo source material at the target cutoff, and report event-level errors, binary loss/calibration, ranking, candidate coverage, and sensitivity across time/source aggregation choices. The inspected 2019 study is development data and cannot serve as an untouched test. With the current small weak-label corpus, the graph pass is primarily a data-quality and representation experiment—not sufficient evidence to fit a large weighting model.

## Entity profiles and query-conditioned GNN compatibility

Profiles are generic (`person`, `organization`, and other entity types), so Trump, Elon Musk, companies, and institutions use the same representation. Store profile facts as `entity_claim` nodes with `valid_from`/`valid_until`, `available_at`, confidence, and citations. Store relations—such as employment, ownership, public association, meeting, or dispute—as typed, time-bounded edges with evidence. Preserve aliases and entity-resolution uncertainty; do not merge entities based only on a shared name.

This structure can later support a **query-conditioned GNN** (Q-GNN-style) without changing the evidence contract:

1. Anchor a target-event query to only its as-of story/entity subgraph.
2. For each proposed idea candidate, condition message passing/attention on that candidate rather than reducing the whole graph to one shared summary.
3. Let typed paths convey evidence, e.g. event → story development → person/entity → dated profile claim or relation → another story → candidate concept.
4. Include time, confidence, source diversity, duplicate-vs-update status, and provenance in node/edge features; retain paths and source citations for explanations.
5. Train and validate chronologically; no target transcript labels, post-cutoff profile facts, or links discovered from future articles may enter that event's graph.

This is a design target, not a claim that a Q-GNN exists or can be trained reliably with current sample sizes. The older parameter-free graph/path experiment in `speechprefix/idea_layer/news_phrase_backtest.py` is not a trainable Q-GNN and is not the implementation target.

## First implementation boundary

The new `story_graph.py` contract and synthetic tests establish typed nodes/edges, source-ledger resolution, content hashes, outcome-field rejection, valid-time checks, strict as-of filtering, and a clearly flagged retrospective mode. Next, build a reversible deduplication/update-candidate extractor over the locally retained 2019 source snapshots and emit cluster-reduction diagnostics. Keep its output separate from the active model until the full pipeline, chronological comparison, and per-event analysis are reviewed.
