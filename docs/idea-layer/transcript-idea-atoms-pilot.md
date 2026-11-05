# Transcript-only semantic idea-atom extraction pilot

**Status:** AI-authored, transcript-only development diagnostic. These outputs are interpretive candidate labels, not semantic gold, and they have not been independently validated. This pilot addresses the 2019 records that the compact lexical extractor left empty; it does not change that extractor or the forecasting model.

## Why this pilot

The compact extractor (`compact-issue-phrases-v2`) only emits short phrases matching a small issue-anchor and curated-concept inventory. Its 49 empty outcomes in the 2019 candidate corpus were not missing transcripts: each had at least two target-speaker turns (median 14), and the available transcript lengths ranged from 38 to 2,849 words (median 704). An empty lexical result should therefore mean “no match from this extractor,” not “the speech had no ideas.”

## Extraction method

A separate open-vocabulary LLM extractor was run on only those 49 weak-label-empty events. For each event, it received the complete available target-speaker turns, with event ID/date/title, news, forecasts, and weak labels withheld. The model produced concise concept atoms, a neutral summary, type/stance/prominence/confidence, an interpretive `forecastable` flag, and one or more exact supporting quotes with turn IDs. Event identities and original turn IDs were mapped back only after generation. The runner checks that every quote is an exact substring of the cited target-speaker turn and rejects unknown turn citations. This checks provenance, not whether the quote semantically supports the concept.

The pilot used Pi CLI **1.4.0**, provider `openai-codex`, model alias `gpt-6-luna`. The model alias is not a pinned immutable checkpoint. Transcript text was sent to that configured inference provider; generated records and reports remain local and Git-ignored. The new extractor is a prompted model workflow, not a locally trained supervised model.

## Results

| Diagnostic | Result |
|---|---:|
| Previously weak-label-empty 2019 events processed | 49 |
| Candidate atoms returned after one failed item was retried | 269 |
| Events with at least one candidate atom | 49/49 |
| Atoms marked forecastable by the model | 190/269 |
| Events with at least one atom marked forecastable | 46/49 |
| Exact evidence quotes validated against cited turns | 440/440 |
| Events with no forecastable atom | 3 |

For this selected subset, candidate-output coverage went from 0/49 lexical labels to 49/49 LLM-generated atom inventories. This is a coverage comparison only; it does not establish that the new atoms are correct.

The initial 49-event run produced atoms for 48 events; one response failed exact-quote validation on both attempts. A separate transcript-only retry succeeded and was merged as a new immutable artifact. The original run, retry, and merged artifact are preserved separately under `data/source-archives/`.

The output distinguishes, for example, a ceremonial award mention (marked nonforecastable) from policy/political concepts. That distinction prevents every transcript mention from automatically becoming a candidate forecasting target. At the same time, the model marked 264 of 269 atoms “high” confidence; that self-reported confidence is uncalibrated and should not be read as quality evidence.

## Interpretation and limitations

This demonstrates broader **candidate coverage**, not improved forecast accuracy or proven label correctness. A quote can be present verbatim while the model's abstraction, stance, prominence, or forecastability judgment is still wrong. The LLM may over-split, merge unrelated propositions, omit concepts, or overstate confidence. No human review was used, and no independent semantic gold exists.

This is a selected slice of 49 events whose weak labels were empty. The other 142 labeled 2019 events and earlier training events have not yet been processed by this extractor, so the 269 atoms cannot be mixed into the old evaluation or compared as if they were a uniform target set. No prediction model was trained or scored against these atoms. Before using them in forecasting, the same extraction protocol would need to be applied to the chronological training and evaluation events, with duplicate concepts aligned across events and per-event disagreements examined. The current `idea_id` values are event-local open-vocabulary labels, not a stable cross-event ontology.

Transcript content may make an event recognizable despite hiding metadata. Forecastability is a model judgment based on the transcript alone, not evidence that pre-event news would have predicted the idea. The extracted atoms are for retrospective outcome representation only; the target transcript remains forbidden as its own prediction input.

## Local reproducibility

The input corpus is `data/rollcall-by-type/trump-speech-candidates.jsonl` (SHA-256 `c7e337830eddf0c1213759f87045369b862d2a74a5c1d89515912f84819a1fdf`). The extraction script is [`../../research/extract_transcript_idea_atoms.py`](../../research/extract_transcript_idea_atoms.py); it requires the Pi CLI and an authenticated compatible model provider:

```sh
uv run python research/extract_transcript_idea_atoms.py \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --output data/source-archives/2019-weak-empty-semantic-idea-atoms-v1.jsonl
```

The retry merge utility is [`../../research/merge_transcript_idea_atom_retries.py`](../../research/merge_transcript_idea_atom_retries.py):

```sh
uv run python research/merge_transcript_idea_atom_retries.py \
  --base data/source-archives/2019-weak-empty-semantic-idea-atoms-v1.jsonl \
  --retry data/source-archives/2019-idea-atoms-retry-fe02b997-v1.jsonl \
  --output data/source-archives/2019-weak-empty-semantic-idea-atoms-v1-complete.jsonl
```

All output paths above are Git-ignored. Recorded hashes:

- Extraction prompt SHA-256: `377d072fbe20faaab52906346ffc1a096ed3c0cc85989f4d163c0a0b56995ece`
- Extraction script SHA-256 used for the pilot: `a738ff88620c05a973b0bf705644a915b85a8e605406ecc968a8090cf6a742e1`
- Initial 49-event artifact: `d8f1aacf761fa129ec57e2266c08f069ebbb290f13d682e8b9dbbb6047b021d0`
- One-event retry artifact: `462a826ec0c83fa03f31e60a2862d90920f19c3ba46f931df51d4e23caefa2d4`
- Merged 49-event artifact: `024d6033f6d9330b14752a2956427c33ab67fe040cb8af5286f34fcb6723e734`
- Merged run manifest SHA-256: `8c22e946c6137b0bf324e249736cc52fe6fa1d48265f5ee920beba43c40568b0`
