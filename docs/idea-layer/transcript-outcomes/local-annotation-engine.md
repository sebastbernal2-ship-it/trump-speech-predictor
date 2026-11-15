# Local annotation engine

See the [offline methods research and experiment order](offline-methods-research.md) for the proposed next iteration.

Codex is a reference annotator only. The frozen development run `annotation-review-v7` contains 66 Codex drafts and one chat-authored draft. These are not adjudicated gold. The local rules engine reads only transcript packets, uses no model API, and never sees reference labels during inference. Both reference and candidate artifacts stay local and outcome-only.

```sh
uv run python -m speechprefix.idea_layer.local_annotation_evaluate \
  --reference-run data/source-archives/speech-content/outcomes/annotation-review-v7 \
  --output data/source-archives/speech-content/outcomes/local-engine-eval-new
```

The evaluator verifies reference hashes and packet identities, scores exact evidence, normalized label text, and labeled edges, and emits per-view matched/candidate/reference counts. Exact agreement means all multiset counts match; it is not a semantic paraphrase metric and ignores notes, confidence, and ordering. Identical node descriptors are interchangeable in the edge comparison.

The initial baseline agreed on **1/67**, an empty reference; **0 nonempty exact matches**, 58 abstentions, and 5/358 exact reference mentions matched. It has a small subject vocabulary and explicit first-person stance cues. It does not yet extract finer details or tone. This result does not meet the development target. Do not run on the remaining corpus as a production system yet. Improve general extraction and use error categories and reviewed semantic matches alongside exact scoring; never add view-ID lookups or copy drafts as predictions. Even 67/67 development agreement would not establish unseen accuracy.

## Six-layer experiments

Use `--engine classical` or `--engine encoder --model-dir /path/to/local/checkpoint` with the evaluator above, choosing a fresh output directory each time. The encoder requires optional PyTorch and Transformers installations and local safetensors weights. Both variants share evidence spans, open-vocabulary phrase candidates, and cue-based detail, stance, and tone extraction; only subject and topic grouping differ (token overlap versus local encoder similarity). Neither sends transcripts to a service.

On the same 67 development references both scored 0 exact full matches and recovered 108/358 exact evidence mentions. Classical grouping produced 373 ideas; the cached DistilRoBERTa checkpoint produced only 67 (282 reference ideas), collapsing distinct subjects. Both produced just two key-point candidates (263 references), with no exact matches. Normalized label agreement is not semantic equivalence. These experiments are not rollout-ready.
