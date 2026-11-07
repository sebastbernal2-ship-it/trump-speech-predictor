# Methods for extracting key conversation points: literature review and design

**Purpose:** identify established methods relevant to building a speech extractor that returns the speaker's key ideas/points—not broad topic labels, and not every sentence-level assertion. This is a method review, not evidence that any cited approach solves this exact task. Primary ACL papers were retrieved and their full PDFs inspected for the claims below. The repository's research-engineering references (`research/README.md` and `research/research_protocol.py`) are absent, so this note records source URLs and claims directly rather than using that missing citation-manifest protocol.

## Relevant established methods

### 1. Argument-to-key-point analysis (KPA): useful point representation, but not automatic point discovery

Bar-Haim et al. define key points as concise talking points and quantify salience by the number of input arguments that map to each point. Their ArgKP study uses **expert-authored key points**: a professional debater generated candidate points from each topic without seeing the arguments, unified related points, and selected up to seven per side. Crowd annotators then mapped arguments to all applicable points or “none”; multiple judgments were aggregated, and ambiguous matches were retained as uncertainty rather than forced into a single class. The expert-written points covered 72.5% of the arguments in their collection.

**Implication:** this is a strong precedent for what a useful point should look like and for evaluating point coverage/semantic matches. It is *not* evidence that an LLM can discover a complete inventory from one speech: the central KPA experiment assumes its point set is supplied by an expert. The present extractor's one-shot generation is attempting the less-established task.

Source: Bar-Haim et al. (2020), [“From Arguments to Key Points: Towards Automatic Argument Summarization,” ACL](https://aclanthology.org/2020.acl-main.371/), [PDF](https://aclanthology.org/2020.acl-main.371.pdf), DOI `10.18653/v1/2020.acl-main.371`.

### 2. Point extraction from political debate: extract proposition-like units, then organize them

Egan, Siddharthan, and Wyner (2016) study online political debates. Their system extracts a short point from a verb and its syntactic arguments, using dependency parses and VerbNet frames to form a reusable point pattern and readable text span. They filter generic/noisy patterns, cluster realizations with the same pattern, rank points by frequency, and produce structured sections for common points, counterpoints, and co-occurring points. Human evaluation preferred these point-based summaries to a sentence-extractive baseline; organization into sections improved preference further.

**Implication:** the pipeline separates *point-unit extraction* from *grouping/organization*, instead of asking one operation to make broad summary labels. Its limitations matter: it is a 2016, syntax-driven method for multi-author forum debates; the shared verb/argument pattern is brittle for paraphrase and political speech, and the paper notes that frequency-ranked summaries can suppress minority viewpoints. We should borrow the stages and explicit point/evidence representation, not copy its parser or treat frequency as completeness.

Source: Egan, Siddharthan, and Wyner (2016), [“Summarising the points made in online political debates,” ArgMining](https://aclanthology.org/W16-2816/), [PDF](https://aclanthology.org/W16-2816.pdf).

### 3. Key-point hierarchies: keep points separate; represent broadness and relatedness explicitly

Cattan et al. (2023) identify a known KPA failure mode: flat lists contain points at differing levels of specificity and related ideas, making them hard to interpret. Their Key Point Hierarchies (KPH) cluster near-equivalent phrasings and connect specific points to broader points through directed entailment/support relations. The hierarchy is built after key points exist, using pairwise relation judgments; THINKP uses consolidated multiple annotations.

**Implication:** do not solve broad-vs-specific errors by merging a detailed idea into a broad topic. Preserve independently meaningful point nodes and add optional broader-theme or support/instance/contrast edges. The extractor's prediction targets should remain the specific points; hierarchy is a navigation/analysis layer.

Source: Cattan et al. (2023), [“From Key Points to Key Point Hierarchy: Structured and Expressive Opinion Summarization,” ACL](https://aclanthology.org/2023.acl-long.52/), [PDF](https://aclanthology.org/2023.acl-long.52.pdf), DOI `10.18653/v1/2023.acl-long.52`.

### 4. Pyramid content units: evaluate against multiple human readings, not one generated list

Nenkova and Passonneau's Pyramid method creates semantic Summary Content Units (SCUs) from multiple human summaries, assigns greater weight to content units supported by more annotators, and scores a candidate summary by the weighted content it covers. Its motivation is that there is no single uniquely correct summary and one reference misses valid content-selection variation.

**Implication:** a high-quality speech reference should be built from independent human readings, then adjudicated; one AI inventory must not become its own gold. A Pyramid-style weighted coverage score can measure whether an extractor captures high-consensus points while retaining valid minority points. Add separate precision/specificity and merge/split diagnostics; Pyramid alone does not certify proposition boundaries.

Source: Nenkova and Passonneau (2004), [“Evaluating Content Selection in Summarization: The Pyramid Method,” NAACL-HLT](https://aclanthology.org/N04-1019/), [PDF](https://aclanthology.org/N04-1019.pdf).

### 5. Topic segmentation: possible support for long speeches, not a substitute for idea extraction

Hearst's TextTiling segments text into multi-paragraph subtopic passages by lexical cohesion. This is a classic boundary-detection method, not a claim/idea extractor. For speeches, speaker turns and transcript paragraphing may already provide usable structure; any automatic segmentation must be evaluated for missed cross-boundary points.

Source: Hearst (1997), [“Text Tiling: Segmenting Text into Multi-paragraph Subtopic Passages,” Computational Linguistics](https://aclanthology.org/J97-1003/), [PDF](https://aclanthology.org/J97-1003.pdf).

## Synthesis: the method to build here

The established literature supports a **staged, evidence-grounded pipeline**, not a one-shot abstract list:

1. **Segment for coverage.** Traverse target-speaker turns in order (use discourse boundaries where available); process manageable contiguous units so long speeches do not disappear into one opaque prompt.
2. **Extract local key points.** For each point, state the actual subject + predicate/claim/action + stance, cite exact transcript spans, and retain surrounding turn context. Keep supporting facts/examples as evidence unless the speaker gives them independent focus. Do not use generic labels such as “Ukraine,” “impeachment,” or “Biden” as the point.
3. **Deduplicate only semantic equivalents.** Across segments, merge only paraphrases that express the same proposition and stance. If identity is uncertain, keep both and flag for review. Shared topic, actor, or event is not enough to merge.
4. **Add a separate organization layer.** Optionally connect points as same-theme, supports, illustrates, contrasts, or specific-to-general. These edges never erase or replace leaf point IDs.
5. **Audit completeness and quality separately.** Check each segment for missed key points, broad topic-only outputs, unsupported claims, and over-splitting. A model audit is a diagnostic; exact quote validity proves provenance only.
6. **Evaluate against independent human references.** Have multiple annotators independently identify points with evidence spans before seeing model output; adjudicate duplicate, merge, and split disputes. Score weighted coverage/recall (Pyramid-inspired), unsupported/irrelevant points, claim specificity, atomicity, evidence faithfulness, and inter-annotator agreement separately. Include short and long speeches and multiple speech types. Do not optimize against a single speech or the old 54-item inventory.

This design produces both a **point inventory** for prediction and a **point hierarchy** for interpretation. Current outputs v2.5, v2.6, and chunked v1.5 are exploratory branches, not training labels. The frozen seven-day baseline and all chronological evaluation remain unchanged.

## Local prototype evidence and known failure

- Existing v2.5 single-pass prompt produced 98 ideas on the 9,012-word Lexington rally, including asides promoted to ideas (over-splitting).
- v2.6 added a salience gate and produced 58; it still merged allegations about different countries into one Biden “foreign dealings” idea (under-specific).
- A new v1.5 chunked point-extraction plus proposition-equivalence dedup run produced 104 ideas. The strict dedup preserved a Ukraine-call dispute separately from impeachment, but the output remained substantially over-split and required further point-level judgment. Its ignored output SHA-256 is `bdd56068a5f3d011f63e2b1c64f2120531a1c030ea428a2a1eced5b47ab6d9c0`; manifest SHA-256 is `08c21e41a0fb086614d266cc611d9e504bebcfe7a3030e8de088291433b30d19`.

These are one-model/one-speech diagnostics, not a valid accuracy comparison. The 98-, 58-, and 104-idea outcomes are preserved locally with model/prompt provenance in ignored manifests under `data/source-archives/2019-11-04-lexington-v*`. No historical circulation analysis should be run from these unreviewed labels.

## Method trials on the Lexington candidates

To test the literature-inspired organization step without altering the original candidate IDs, `research/idea_layers/points/classify_speech_key_points.py` adds a second-pass annotation: `key_point`, `supporting_detail`, `aside_or_ceremonial`, or `uncertain`; optional `specific_to` edges preserve a key-point hierarchy, and evidence/example/qualification edges attach subordinate details. It is deliberately non-destructive: no candidate can be merged, rewritten, or dropped. Structural validation checks full ID coverage, allowed parent types, and cycles. As with all same-provider judgments, this is not independent semantic evaluation.

The same Codex model (`openai-codex`, `gpt-6.1-sol`) classified both the v1.5 chunked inventory and v2.6 single-pass inventory:

| Candidate inventory | Input ideas | Key points | Supporting detail | Aside/ceremonial | Uncertain | `specific_to` edges | Other hierarchy edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chunked v1.5 | 104 | 95 | 4 | 5 | 0 | 2 | 4 |
| Single-pass v2.6 | 58 | 54 | 2 | 2 | 0 | 2 | 1 |

The point-selection pass barely reduces either inventory. In v1.5 it marked the dog visit, a campaign anecdote, and several ceremonial/personal items as non-key, but retained nearly every candidate as a key point. This **does not solve the over-splitting problem**. The hierarchy pass did produce a few plausible broad-to-specific or evidence links (for example, coal/jobs under the broader energy claim), but a model-generated edge is not verified entailment. It kept Ukraine and impeachment separate in this run; that is a useful constraint check, not proof of correct boundaries. A repeat v2.6 classification retained the same status counts but changed the hierarchy: an earlier call returned two support/detail edges and no `specific_to` edge, while the preserved rerun returned two `specific_to` edges and one support edge. This same-provider instability further rules out treating the relations as validated.

A KPA-inspired occurrence diagnostic on the chunked output found that 102 of 104 ideas had only one linked `expresses` mention (two had two). This does not imply they are unimportant: the chunk-wise extractor does not globally align repeated paraphrases, and KPA's argument-frequency signal was developed for a pool of arguments mapped to expert-authored points, not repeated mentions by one speaker. Treat these counts as evidence that the KPA metric cannot be copied into this setting without changing the unit of analysis.

The corresponding key-point classification artifacts are local/Git-ignored. Output SHA-256 values: chunked-input classification `5e8b7797681c154bceace7e45c4897660777dae8a8f500a273fe2a31daef381a`; v2.6 classification `249a58e85121272e82e978a596bb1f246dfbfd175764eda12f31996cadecd782`. Each has a separate manifest with input/output hashes, provider/model, and prompt version. The Pyramid-style human-reference evaluation remains **not run**: no independent human annotations are available, and substituting this model's second pass would be circular. A real trial requires independent annotators and adjudication before candidate outputs are shown to them.
