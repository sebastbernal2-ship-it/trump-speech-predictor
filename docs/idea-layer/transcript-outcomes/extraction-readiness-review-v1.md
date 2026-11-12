# Transcript extraction readiness review v1

## Finding

The current output is useful for locating **unreviewed lexical candidates**, but is not ready as a complete set of supervised-learning targets. A successful run and traceable spans do not establish that labels are correct, complete, attributable to the intended speaker, or consistently defined. Phase 1 is not demonstrated complete for all speeches or layers.

## Scope and method

Reviewed six deliberately varied records from the local `speeches.jsonl` view and their `transcript-candidates-v1` outputs: source lines 222, 291, 501, 968, 969 and 1019. These include short statements, a meeting and an exchange with reporters. The four short records were read in full; the two longer records were inspected in part, including candidate evidence. This is a diagnostic convenience sample, not a representative random sample or a semantic precision/recall study. The IDs and line numbers below identify records in that particular source view; they do not establish unique events across all available views.

Compared the raw source and cleaned text and examined candidate labels and passages. Separately checked all 1,019 paired source/output rows: IDs and raw SHA-256 hashes matched, the recorded cleaned text matched the current whitespace cleaner, and all 89,005 candidate evidence spans matched their clean slices and their whitespace-normalized raw slices. These are structural checks, not independent semantic validation; pairing by line does not establish corpus completeness or event identity.

## Observations against source

| Source line and ID | Observation |
| --- | --- |
| 222, DCPD-201700382 | The source mourns victims of attacks in Iran and says states sponsoring terrorism risk becoming victims. `Iran attacks` has relevant evidence and `terrorism` is present, but neither captures the latter specific claim or its stance. The first evidence passage merges the heading, date and statement after whitespace cleaning. |
| 291, DCPD-201700478 | The statement sends well wishes to Senator McCain about his health. The only topic is `family`, and there are no idea candidates; the main health and well-wishing content is missed. The period in `John S.` also fragments the title and metadata into passage boundaries. |
| 501, DCPD-201700769 | The meeting discusses the United Nations and its potential, but has no idea candidates. `right` comes from “right now,” and `job` from praise for a person's work, illustrating ambiguous issue anchors. `world` is supported by Secretary-General Guterres's words, so cannot automatically be attributed to the President. |
| 968, DCPD-202500433 | The short statement praises job numbers and says the policy is working. `job` is detected, but no idea captures the claim. The heading and date remain in its evidence. |
| 969, DCPD-202500432 | The statement addresses investors and says policies will not change. `economy` is matched from the document title, while no idea is produced. A title-derived topic is not proof that the word or concept was actually spoken. |
| 1019, DCPD-202500575 | The source contains reporter questions, editorial headings, annotations and an incomplete opening. `Saudi Arabia` is supported by a reporter's question, and `agriculture` by the editorial Subjects field; neither establishes that the President discussed those items. `right` is matched in “right now.” The opening remarks about an American Pope are a salient idea that is not captured by the two idea candidates. |

The cleaner consistently collapses whitespace in this view, but does not isolate spoken text or remove editorial material. It cannot recover inaudible or missing speech. Source text includes formal written statements as well as remarks and exchanges, so inclusion in the target population needs an explicit decision. The example labels are diagnostic judgments, not a complete annotation of each speech.

## Coverage and readiness limits

The manifest reports 1,019 records, 18,823 topic candidates and 5,856 idea candidates; 7 records have no topic matches and 187 have no idea matches. All records have nonempty cleaned text. Counts describe this one selected flattened view and its lexicon, not all available transcripts, distinct speeches, true absences, or semantic coverage. Duplicate and overlapping source views, source quality, date and speaker identity, and any missing records remain unreconciled. No independent reference was constructed for this review, so there is no defensible precision, recall, inter-annotator agreement, or readiness threshold here.

The mechanical layers also need a population and source-text decision before they can be considered clean targets: exact characters and words depend on which text is included; period splitting is the requested mechanical rule, not a guarantee of linguistic sentences, as abbreviations in line 291 demonstrate. The structural checks above cover candidate cleaning and evidence, not a fresh end-to-end validation of mechanical outputs for every source.

## Decisions and remaining work

1. Inventory and reconcile the intended complete event population and overlapping versions; document exclusions, missing or incomplete transcripts, source rights, and stable identities. Decide whether statements, reporter exchanges, other speakers, headings, annotations and metadata belong in each target layer.
2. Specify versioned topic and idea units, granularity, boundaries, deduplication and relationships, and how to record uncertain, absent and incomplete cases. Decide whether and how stance or degree is annotated; text alone does not establish vocal tone.
3. Establish a source-backed cleaning and speaker-attribution procedure with traceability to raw text, and validate character, word and period-split sentence outputs on the selected population. Preserve raw text and distinguish missing or partial content from true absence.
4. Build independently annotated references on a documented, varied sample, including no-match and multi-speaker records. Measure candidate precision and recall against those references, agreement and error types, adjudicate disagreements, and set quality and coverage thresholds before accepting targets. Resolve incomplete or failed records explicitly rather than interpreting them as negative labels.
5. Produce versioned, provenance-linked target datasets for every intended layer and account for every in-scope event. Reassess readiness after the above checks; no claim about later prediction performance follows from this review.
