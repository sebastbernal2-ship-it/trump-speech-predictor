# Offline six-layer annotation: methods and experiment order

Recommendation: parse → propose spans and nodes → rank explicit links → validate. Use a shared candidate bank, a sparse classical comparator, and small task-specific local encoders. No text-generation model, API, or transcript upload at inference. Do not use annotations as same-event forecast features.

| Layer | Start with | Compare / upgrade |
|---|---|---|
| Evidence | Preserve immutable turn text and character offsets. Propose whole sentences and dependency-delimited clauses, including coordinated propositions. Keep context turns for interpretation only. | Compare sparse span ranking with an encoder start/end or span ranker; pretrain on [presidential debate claims and premises](https://aclanthology.org/P19-1463/) only as auxiliary supervision. Measure candidate recall before ranking; use [ERASER](https://aclanthology.org/2020.acl-main.408/)-style token overlap and exact spans. |
| Ideas/occurrences | Use noun chunks, named entities plus issue heads, and predicate–argument frames as open-vocabulary candidates. Build extractive canonical labels from supported source phrases. Keep occurrences distinct by actor, place, date and event type. | [Stanza dependencies](https://aclanthology.org/2020.acl-demos.14/), [MAVEN](https://arxiv.org/abs/2004.13590) for event triggers, and [DyGIE++](https://aclanthology.org/D19-1585/) as a span-model design reference. Event detection alone misses issues and subjects. Benchmark conservative coreference ([Maverick](https://aclanthology.org/2024.acl-long.722/)) separately. Compare sparse retrieval and [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5) for neighbor proposals, then context-aware pair scoring with hard negatives. Never merge solely on cosine similarity. |
| Key points | Propose event arguments, named subevents, measures, dates, quantities and subordinate propositions from syntax/SRL. Keep only substantive refinements; attach a child to a parent only when passage and relation support it. | Train a detail selector and a parent-link ranker, not a generic summary model. Compare sparse dependency/lexical features with a context-aware encoder on (parent, candidate detail, passage). Include same-topic wrong-parent negatives. Measure oracle-parent and end-to-end results separately. |
| Topics | Classify idea plus its evidence into multiple broader policy categories using [CAP](https://compagendas.la.utexas.edu/pages/master-codebook) as optional scaffolding; keep non-policy and out-of-taxonomy cases. | Compare word/character TF-IDF plus logistic regression versus a small multilabel encoder. Offer extractive open-topic grouping as fallback. Do not substitute taxonomy labels for individual occurrence IDs. |
| Opinions/stances | Predict speech act (assertion, proposal, request, forecast) separately from targeted support/opposition; map to the artifact with an explicit mapping. Attribute the claim and represent reported speech, negation, hypotheticals, and uncertainty. | Use a passage–target cross-encoder; [VAST](https://aclanthology.org/2020.emnlp-main.717/) can warm-start polarity, not speech acts. [ClaimBuster](https://zenodo.org/records/3609356) provides auxiliary claim detection, not truth or all speech acts. Compare an NLI checkpoint such as [DeBERTa-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) as an initialization or ablation, not an authority. |
| Emotion/tone | Score only text-supported rhetorical tone; include unclear/abstain. Do not infer private emotion or audio prosody. | Compare lexical baseline and a multilabel local classifier, transfer from [GoEmotions](https://aclanthology.org/2020.acl-main.372/) only as a starting point. Its Reddit labels do not directly represent rhetorical functions such as condemnation or mockery. [Parliamentary-speech emotion research](https://aclanthology.org/2024.parlaclarin-1.11/) supports domain adaptation as an experiment, not an English checkpoint recommendation. |

## Benchmark before rollout

1. Freeze the 67 draft files and evaluator configuration; retain exact graph agreement as a regression diagnostic, not the only metric. Inspect a reviewed slice to establish span boundaries, label equivalences, and positives/negatives for target links. Never patch based on transcript IDs or copied reference text.
2. Run an offset-preserving candidate-recall bake-off first: sentence-only vs sentence + dependency clauses, then add named-event arguments. Both current variants matched 108/358 exact evidence mentions and produced 2/263 key-point candidates. Report recall at fixed candidate budgets, exact and token-overlap metrics, and failure examples by length/format. Do not optimize classification before checking its candidate ceiling.
3. Compare sparse and encoder rankers on the same candidates. Use event-grouped splits that keep alternate views and near duplicates together. Score aligned ideas and details with one-to-one matching and reviewer-checked paraphrases; report split/merge errors and typed edge F1 after alignment. Report stance and tone macro-F1 plus coverage vs error under abstention. Report results both with reference inputs per stage and end-to-end.
4. Record model revision, license, local weight hashes, CPU throughput/RAM, and coverage; run with offline model loading and explicit weights. Set acceptance thresholds before examining a separate reviewed holdout with long transcripts, different formats and unseen subjects. The 67 drafts are development supervision, not semantic gold; even 67/67 cannot demonstrate unseen accuracy. Do not roll out on structural validity alone.

These sources motivate the component choices; no new component was benchmarked in this research change. The first engineering experiment is the candidate-recall bake-off, then parent-linked detail extraction. Fixing the encoder similarity threshold alone cannot recover missing details or teach stance or tone.

## First candidate benchmark

Run `python -m speechprefix.idea_layer.annotation_candidate_eval --reference-run PATH --output NEW.json` in an environment with spaCy 3.8 and the locally installed English `en_core_web_sm` 3.8.0 pipeline. Model installation is separate; the evaluator does not contact a provider. Use an ignored output path.

On 67 development views, sentence-only proposals hit 157/410 unique turn/quote references exactly and 288/410 at character IoU ≥ 0.5. Adding parsed sentences and clauses raised these to 178 and 342; adding arguments and entities raised them to 196 and 362. The pool grew from 473 to 1,628 unique offsets (summed per view). For key-point-linked evidence, exact hits rose from 119/321 to 149/321 and overlap hits from 225 to 282. These are oracle proposal recall counts, not precision, semantic detail extraction, or end-to-end scores. The previous 108/358 count scores complete mention nodes under a different denominator.

## Event-grouped span ranking

Run `python -m speechprefix.idea_layer.annotation_span_rank --reference-run PATH --output NEW.json --budget 8` with the same parser and scikit-learn (tested 1.9.1). Five event-ID-grouped folds train sparse logistic rankers on same-turn character IoU >= 0.5 labels. This is development cross-validation, not a near-duplicate audit or an independent holdout.

At the same budget (480 selected spans summed across 67 views), learned versus sentence-first ranking hit 237 vs 246 of 410 evidence references by overlap (119 each exactly). For key-point-linked evidence it hit 172 vs 178 of 321 by overlap and 85 vs 81 exactly. Keep sentence-first as the overlap baseline. Detail-evidence ranking does not identify the detail itself or its parent; test parent-link candidates and ranking separately. This diagnostic does not replace production extraction or forecasting.

## Detail and parent-link diagnostic

Run `python -m speechprefix.idea_layer.annotation_detail_link --reference-run PATH --output NEW.json` with the same local parser. On the 67 drafts, 1,144 argument/entity/clause proposals yielded zero normalized full-text detail matches with supporting evidence. This strict score does not measure paraphrase equivalence. With reference details, evidence and idea labels supplied, the lexical linker chose the correct parent for 210/263 details (240 assigned) and retrieved it in the top three for 239/263. This isolates linking under oracle inputs; it is not end-to-end performance. Next, compare extractive candidates and reference paraphrases with calibrated semantic matching before deciding whether to expand or rank the span pool.

## Semantic detail review queue

Run `python -m speechprefix.idea_layer.annotation_detail_review --reference-run PATH --output NEW.json` with the local parser. It ranks literal spans by lexical overlap inside reference evidence windows and exports both syntax-only and sentence-inclusive lists for review. Scores are retrieval cues, not semantic verdicts. On the 67 drafts, 242/263 points had a syntax candidate overlapping their evidence, versus 263/263 with sentences included. Spot checks found cases where full sentences preserve the assertion that a bare noun phrase loses, and cases where neither list recovers the full detail. Presence in the queue is not recall of equivalent details.

A seeded 16-point assistant spot audit (seed 73, top two per list, partial/uncertain excluded) found full detail support for 2/16 syntax-only and 13/16 sentence-inclusive candidates. Adding adjacent same-turn sentence pairs (maximum 60 words) also scored 13/16; do not promote those pairs by default on this evidence. This small diagnostic uses reference evidence windows and is neither adjudicated nor end-to-end.

## Parent-conditioned sentence selection

`python -m speechprefix.idea_layer.annotation_detail_selector --reference-run PATH --output NEW.json` compares lexical ranking by occurrence label with chronological selection. At two spans per idea (559 proposed links per method), lexical ranking found overlapping key-point evidence for 137/263 points versus 63/263, and exact evidence for 65 versus 31. Only idea labels are supplied to the selector; reference details and quotes are used for scoring. This still measures retrieval under oracle ideas, not semantic detail equivalence or production accuracy.

## Transcript-only detail integration

`python -m speechprefix.idea_layer.annotation_detail_integration --reference-run PATH --output NEW_DIR` attaches extractive sentences to locally extracted ideas before the frozen drafts are read by the scorer. On the 67 drafts it produced 521 detail candidates versus 263 reference details, with zero exact detail or parent-edge matches and zero exact graph matches. This increases proposal volume, not demonstrated accuracy. Keep it experimental; evaluate semantic matches and parent identity before rollout.

## Idea agreement beyond exact wording

The evaluator reports one-to-one `subject_alignment` separately from exact graph agreement, with unmatched IDs for review. It accepts Russian–Ukrainian War and Ukrainian war vs Russia as the same occurrence, without dropping event qualifiers or reversing directional relationships. This conservative alias check is a floor, not a general paraphrase scorer. It leaves evidence validation unchanged. On the 67 development drafts, the integrated engine still aligned only 1 of 282 reference ideas (373 proposed); unmatched labels are not automatically semantic errors.

`python -m speechprefix.idea_layer.idea_pair_review --reference-run PATH --candidate-run PATH --output NEW.json` queues unmatched idea pairs with quotes and explicit blank verdicts. The v2 detail integration run produced 717 pairs for 280 of 281 unmatched reference ideas. A shared turn or overlapping label is only a retrieval signal. Spot checks showed both wording differences and vague labels that lose the subject, as well as reference ideas phrased as assertions. Never infer semantic accuracy from queue coverage.

## Parsed idea labels

`python -m speechprefix.idea_layer.parsed_annotation --reference-run PATH --output NEW_DIR` replaces discourse-marker bigrams with ranked noun chunks when available. On the 67 drafts it aligned 2/282 reference ideas under the conservative alias scorer (359 candidates), versus 1/282 (373 candidates) for the classical baseline. No broad semantic improvement is established. Bare noun phrases still miss event relations, and unparseable spans retain the old fallback.

## Actor–action–object labels

The parsed evaluator has an opt-in `--frames` ablation. Selected active verbs with an explicit noun-phrase subject and object yield a nominal occurrence label; the quoted source and speaker stance are unchanged. On the frozen 67 drafts it changed only one idea label and left conservative alignment at 2/282. It is not a general event extractor or rollout candidate.

## Reviewed subject-pair diagnostic

An assistant inspected the top two queued pairs for a seeded sample of 16 unmatched ideas (seed 55). Five conveyed the same subject, two were broader/narrower, six were different or lost the subject, and three had an unresolved subject or a questionable layer assignment. Labels must convey the subject; citing the same sentence does not rescue a generic label. This is a small, non-adjudicated shortlist audit of the classical detail-integration run, not an overall semantic score or an evaluation of the later event-frame variant. The frozen drafts remain unchanged.

The opt-in `--grounded` experiment adds predicate–argument phrases and bounded noun complements. On the same 16-point diagnostic it matched 5/16, unchanged in aggregate: it recovered the debt and meeting subjects but lost other matches. The full conservative score was 1/282. Keep experimental; unresolved references and tone-only references were not counted as successes.

The idea-pair review CLI accepts repeatable `--alternative-run` inputs to pool grounded, parsed, and classical hypotheses without replacing their original outputs. The fixed 16-item assistant audit found a same-subject proposal in the top two for 7/16, versus 5/16 previously. This is a reference-assisted shortlist diagnostic, not an inference-time selection score.

## Paired six-layer feedback

Run `python -m speechprefix.idea_layer.annotation_feedback --reference-run PATH --candidate-run RUN_A --candidate-run RUN_B --output NEW.json`. Reports bind inputs by hash, show exact counts, and queue evidence-grounded comparisons for all six layers and typed links. Treat missing cited-turn candidates, same-turn candidates, and quote containment as routing cues; assess alignment before judging links. A connected shortlist is not a correct graph edge. Reviewers should mark same meaning, broader/narrower, different, or unclear; then compare methods on fixed reviewed cases and a separate holdout.

Across the 67 drafts, the integrated run had quote containment for 230/263 detail nodes vs 2/263 in the grounded-label run, but still had zero exact detail matches. Both lacked a same-turn candidate for 67/265 opinions and 43/66 tone nodes. The integrated run had no shortlisted endpoint for 67/269 opinion links, and no link between shortlisted endpoints for a further 37. Prioritize stance/tone candidate coverage and reviewed subject/detail alignment; these diagnostics do not establish semantic recall or authorize rollout.

## Scoring reviewed graph alignments

Run `python -m speechprefix.idea_layer.annotation_review_score --reference-run PATH --candidate-run PATH --review REVIEW.json --output NEW.json`. The review binds the reference manifest and each reviewed candidate graph by SHA-256. Each `records` entry has `view_id`, `candidate_sha256`, and `judgments` entries with `layer`, `reference_id`, and `candidate_id` (null for reviewed but unmatched; omit for unreviewed). A candidate can match at most one reference node per layer per view. For each relation the report separates unreviewed links, unmatched endpoints, missing links between aligned nodes, and correct typed links. These are conditional counts, not precision or full graph recall.

Smoke evaluation used the 13 scorable judgments from the fixed 16-item shortlist audit, with unique exact evidence alignments on those views: five ideas matched. Of 359 reference idea–mention links, 353 had unreviewed endpoints; of the six scorable links, three matched and three had unmatched endpoints. Shortlist nulls indicate no accepted match in the inspected shortlist, not exhaustive misses. The three ambiguous or layer-boundary items were left unreviewed. This does not change the fixed-sample target or authorize rollout.
