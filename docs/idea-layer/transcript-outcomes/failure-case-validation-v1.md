# Source-linked failure-case validation v1

This diagnostic revisits the six records in [the original review](extraction-readiness-review-v1.md). The local versioned [review artifact](../../../.local-reconciliation/failure-case-review-v1/review.json) identifies the exact source file and line hashes, selected and excluded offsets, provisional idea evidence, processing versions and uncertainty. The source is `speeches.jsonl` in the local data root; its path and hash are in the artifact. Reproduce into a **new** output path with `PYTHONPATH=. python3 scripts/validate_semantic_failure_cases.py /path/to/speeches.jsonl /path/to/new-review.json`. The original and earlier versioned outputs are unchanged. Source paths and records are local and have not been cleared for distribution.

The after column is a partial manual diagnostic using `semantic-targets-v2`, not automatic extraction or a complete corrected inventory. Only explicitly selected spans are processed; all remaining text is marked uncertain, including metadata and unreviewed speakers. Therefore an omitted label is not a reviewed negative. The labels and their linked quotes were checked in context by this reviewer, not independently adjudicated. Source labels are evidence of attribution, not verification against audio.

| Source line, ID | Before | After and remaining limitation |
| --- | --- | --- |
| 222, DCPD-201700382 | Iran attack label; specific claim about terrorism sponsors missed; evidence mixed in heading/date. | A provisional idea cites the sentence that sponsors risk victimization; selected text starts at “We grieve”, after the heading/date. The mourning idea and the rest of the inventory remain unreviewed. |
| 291, DCPD-201700478 | Health and well wishes missed; title and abbreviation polluted passages. | A provisional well-wishing idea cites the statement through “Get well soon.” The heading/date are outside the selected span; the title's `S.` is not speech. Full inventory and status as an eligible speech remain unresolved. |
| 501, DCPD-201700769 | `right` from “right now”; `world` supported by another speaker; no idea candidates. | The President's later UN-potential passage supports a provisional idea; the Guterres passage containing “messy world” is context, not target evidence. `right` and `world` are not promoted. Other turns, including the earlier “right now”, are uncertain rather than comprehensively classified. |
| 968, DCPD-202500433 | `job` matched but the job-number/policy claim was missed, with heading/date in evidence. | A provisional idea cites both the better-than-expected numbers and “It's already working.” The selected statement begins after heading/date; no full inventory is claimed. |
| 969, DCPD-202500432 | `economy` came from the title, and no idea was produced. | A provisional idea cites the promise to investors that policies will not change; title-derived `economy` is not promoted. This does not establish that economic discussion is absent. |
| 1019, DCPD-202500575 | Saudi Arabia from a reporter, agriculture from Subjects, and `right` from “right now” contaminated candidates; American Pope idea missed. | A provisional American Pope idea cites the opening remarks. The Saudi Arabia question is context, not direct target evidence; Subjects and all other unreviewed portions remain uncertain. The opening is explicitly joined in progress and includes an inaudible annotation, so this is not a clean complete speech selection. No conclusion is made about whether other answers discuss agriculture or Saudi Arabia. |

For every selected target span, the script checks character reconstruction, exact whitespace-delimited word surfaces, literal `split('.')` including empty/trailing pieces, all unit offsets and clean-to-raw character mapping. It checks each semantic evidence quote against its exact raw source slice and that `build` accepts it only within a target segment. These checks establish mechanical and citation integrity, not semantic entailment or attribution. Short statements and a selected UN passage include periods; the literal split rule deliberately fragments abbreviations and decimals where present. Existing source-linked [speaker selection](../../../.local-reconciliation/revised-run-v1/README.md) separately demonstrates a reviewed question/answer pair.

All six inventories have `uncertain` status and are **not released as training labels**. The selected spans are development examples; partial inspection, missing or inaudible text, written-statement eligibility, source attribution, semantic omissions, unsupported specificity, boundary decisions, independent annotation and adjudication remain unresolved. Fixing these examples does not validate generalization, corpus coverage, or readiness. The frozen benchmark is unchanged.

## Check results (rerun 2026-11-17, `failure-case-review-v1`)

The script asserts every check and aborts on any failure, so a completed run means all passed. Rerunning on the unchanged `speeches.jsonl` gave a byte-identical review artifact.

| Check | Cases checked | Result |
| --- | --- | --- |
| Character reconstruction (joined characters equal clean text) | 6 cases, 6 target spans | 6/6 pass |
| Exact-word surfaces (`\S+` tokens equal word layer) | 6 spans | 6/6 pass |
| Literal period split (`clean.split('.')`, empty/trailing pieces kept) | 6 spans | 6/6 pass |
| Unit offsets and clean-to-raw character mapping | 6 spans | 6/6 pass |
| Idea evidence quote equals exact raw source slice | 6 ideas, 6 evidence spans | 6/6 pass |
| Context-only spans (Guterres, 501; reporter question, 1019) excluded from targets | 2 spans | 2/2 excluded |

Each case produced 1 provisional idea; none is released (`training_label_status: not_released`).

## Unresolved cases

- 222 (DCPD-201700382): mourning idea and remaining inventory unreviewed; 2 uncertain segments.
- 291 (DCPD-201700478): eligibility as a speech (written statement) and full inventory unresolved; 2 uncertain segments.
- 501 (DCPD-201700769): earlier "right now" turns and other turns unclassified; 3 uncertain segments; speaker attribution not verified against audio.
- 968 (DCPD-202500433): no full inventory; whether other economic claims appear is unknown.
- 969 (DCPD-202500432): absence of other economic discussion is not established.
- 1019 (DCPD-202500575): opening is joined in progress with an inaudible annotation; agriculture and Saudi Arabia in answers are unreviewed; 3 uncertain segments.
- All: no independent annotation or adjudication; entailment of quotes not tested; 9,639 of 9,642 inventoried source-view records remain blocked pending source or speaker review (per the revised-run readiness ledger).

## Legacy lexical candidates against the reviewed target spans

`scripts/compare_legacy_candidates.py <transcripts.jsonl> <review.json>` classifies each of the 21 legacy topic/idea candidates for the six records (`transcript-candidates-v1`) by whether its evidence lies in the reviewed target span (output kept locally as `legacy-candidate-comparison.json`). "Outside target" means unreviewed or excluded text, **not** a confirmed false label; "in target" means only that the lexical hit sits in reviewed target text, **not** that the label is correct.

| Line | Legacy candidate (evidence spans: in / partly in / outside target) | Reading |
| --- | --- | --- |
| 222 | attack (0/1/2), Iran attacks (0/1/0), terrorism (1/0/0) | The heading/date-merged evidence is now outside the target; `terrorism` hit is the sponsor sentence, which the provisional idea cites. Stance of the idea is not annotated. |
| 291 | family (1/0/0) | Hit is "their entire family" in the well-wishing; a literal mention, not the main content, and it stays unpromoted. |
| 501 | job (0/0/2), right (1/0/2), world (1/0/2), peace (1/0/0) | Guterres/"right now" hits fall outside the target. Two in-target hits need review, not promotion: `right` ("doing right now" is temporal, still a false lexical match) and `world` ("world peace" is the President's own wording, so the earlier Guterres-only objection no longer applies to that hit). |
| 968 | job (0/1/2) | Heading/date portion excluded; job-numbers claim is carried by the idea, not the word. |
| 969 | economy (0/1/1) | Title-derived hit lies outside the target; one hit partly overlaps. |
| 1019 | 11 candidates incl. Saudi Arabia, agriculture, trade, immigration, election, deal (0 in target; 21 spans outside) | All lie in reporter questions, Subjects or unreviewed answers; they are unreviewed, not rejected. |

## Failure categories not covered by the six cases

The six hand-selected spans cover one example each of: metadata merged into evidence (222, 968), title/abbreviation pollution (291), title-derived topic (969), other-speaker evidence (501, 1019), editorial Subjects (1019), and one omission per record. They do **not** establish:

- **Unsupported or ambiguous lexical labels at scale.** No automatic fix exists: `topics` are empty in every review, and the legacy lexicon was not changed. `right` ("right now") still matches inside reviewed target text (501). The 89,005 legacy evidence spans in the 1,019-row view were checked only structurally earlier, not re-run against revised targets.
- **Semantic omissions beyond one idea per case.** Only one provisional idea per record was added; the remaining inventory of each is unreviewed, so omission rates are unmeasured.
- **Speaker boundaries in turn-based records.** Only 2 of 6 cases (501, 1019) contain a non-President speaker; no automatic speaker selection was exercised on the 7,423 turn-based records.
- **Stance/degree** of any idea.
- **Generalization:** zero cases beyond the six; no random sample, no independent annotation, no adjudication. Reviewed examples covered by these checks: 6 records, 6 target spans.
