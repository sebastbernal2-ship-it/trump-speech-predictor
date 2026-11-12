# Open-vocabulary context → idea generation (Codex prototype)

**Status:** bounded four-event comparator; not a replacement for the frozen seven-day classifier, not calibrated, and not demonstrated stronger. Its purpose is to address one known closed-vocabulary weakness: the current classifier cannot propose an idea label absent from earlier paired training examples.

## Prototype

`research/idea_layers/prediction/generate_open_vocabulary_context_ideas.py` asks Codex to generate up to 15 concrete candidate ideas directly from each event's eligible pre-event headline titles. The prompt receives no event ID/date, target transcript, weak labels, or future sources. Each candidate must cite one or more supplied headline IDs; the local artifact resolves these IDs to source title/date/URL records. Same-day and future headlines are excluded, and the input is bounded to the preceding seven days to match the baseline horizon. The model returns ranked text candidates, not probabilities.

The target event corpus is loaded only to score weak lexical labels after candidate generation. Those labels are not in the model prompt. This exact-phrase comparison is a mechanical diagnostic only: semantic paraphrases, incomplete target extraction, and issue-phrase tokenization cause false negatives/positives.

Run the explicit bounded comparator with a new local output path:

```sh
uv run python research/idea_layers/prediction/generate_open_vocabulary_context_ideas.py \
  --events data/rollcall-by-type/trump-speech-candidates.jsonl \
  --snapshots data/rollcall-context-snapshots-retro-combined-v1.jsonl \
  --output data/source-archives/codex-open-vocabulary-context-ideas-NEW.jsonl \
  --event-id dt47-25-8f06eceb \
  --event-id dt47-25-b560e340 \
  --event-id dt47-26-f641fd3a \
  --event-id dt47-26-5403939c \
  --provider openai-codex --model gpt-6.1-sol
```

The output is local and Git-ignored. The snapshot corpus was retrieved retrospectively; earlier article publication dates do not prove the articles were available in the live input stream at the historical cutoff.

## Initial Codex run

The four events are the existing recent-context development/test slice, not a new untouched benchmark. `openai-codex / gpt-6.1-sol` generated 15 candidates per event (60 total) from 27, 67, 145, and 106 eligible deduplicated headlines respectively. The candidates cited input headlines and were kept open-vocabulary.

Against the weak lexical labels, canonical phrase extraction from candidate text found only **1 of 45** target IDs across the four events (one `iran war` match); the frozen classifier's top 10 contained 16/45 exact-ID hits on the same development slice. This is **not a valid semantic comparison**: the Codex output contains paraphrased, detailed phrases that the lexical matcher cannot normalize, while the training labels themselves are incomplete and sometimes over/under-generalized. Conversely, the Codex candidate lists contain many specific claims/actions that may over-infer what the speaker would say from headlines alone. No independent semantic candidate matcher or human review was applied. The result does not show that Codex is stronger; the generator is a promising open-vocabulary mechanism whose precision/coverage are unmeasured.

Per-event eligible-headline / generated-candidate counts were 27/15 (`dt47-25-8f06eceb`), 67/15 (`dt47-25-b560e340`), 145/15 (`dt47-26-f641fd3a`), and 106/15 (`dt47-26-5403939c`). Because the prompt is capped at 15, it may omit valid ideas; because it returned 15 every time, its effective abstention behavior is not established.

Local artifact hashes:

- Predictions: `data/source-archives/codex-open-vocabulary-context-ideas-v1.jsonl` — SHA-256 `c349f0a9c76f1a1d02de2117f73a2e756b5f1e17a218b456abb927686b17ca69`
- Manifest: `data/source-archives/codex-open-vocabulary-context-ideas-v1.jsonl.manifest.json` — SHA-256 `804c3b8eef651b7eaee3c60e20beb2bb7fb5dda74ff81cf11ed5d428664b0a31`

Full title lists and generated phrases were sent to the configured OpenAI Codex provider. Model aliases are mutable. No output rows, transcripts, headlines, or generated candidates are committed.

## Next work to improve, not merely demonstrate, generation

1. **Measure candidate grounding and unsupported specificity.** Build a blinded audit prompt that compares each generated idea to only its cited headline(s), with verdicts for direct support, plausible but underdetermined, and unsupported. These remain same-provider diagnostics, not accuracy.
2. **Use a separate candidate-matching protocol** to map generated ideas to frozen target ideas without seeing source headlines or model ranks; report disagreements rather than optimizing against a model judge.
3. **Add a learned candidate reranker** over open-vocabulary ideas and source evidence, trained only on earlier chronological event pairs. Keep the current Bernoulli model unchanged as baseline. Compare on rolling chronological folds and report candidate coverage separately from ranking.
4. **Replace the fixed 15-item cap only after validation.** Select candidate count/abstention on earlier chronological validation dates, not the four-event diagnostic slice.
5. **Expand and repair the paired context/outcome corpus**, while preserving source availability uncertainty and target-transcript leakage boundaries. Four events cannot support a robust conclusion.

This prototype advances open-vocabulary candidate generation but does not yet improve the project’s measured forecasting performance. Keep its candidates out of active prediction use until source-grounding, semantic comparison, and chronological evaluation are substantially stronger.
