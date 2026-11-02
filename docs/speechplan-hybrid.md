# SpeechPlan-H

SpeechPlan-H is a from-scratch, word-level model. It does not load GPT-2 or Stanford NCM weights. It uses the installed Transformers GPT-2 implementation only as a randomly initialized autoregressive decoder.

## Inputs and architecture

The model keeps three inputs separate:

- **Transcript prefix:** observed transcript text before the target. A context encoder summarizes it; the decoder also reads it through causal self-attention.
- **Question:** an explicit reporter question, encoded in its own channel.
- **Evidence:** dated, source-attributed text retrieved as of the prediction time. This channel is optional and is accepted only for checkpoints trained with question-linked evidence pairs.

The NCM-like planner combines summaries from those channels and predicts coarse-to-fine residual VQ codes for the next segment. The GPT-style decoder cross-attends to question/evidence memory and the predicted plan codes. A training-only future encoder and residual vector quantizer learn discrete codes at 1, 4, 16, and 32 positions over a 32-token target.

Training mixes ordinary continuation chunks with explicit question-to-president response chunks. The sampler balances the two tasks. The chronological split is by whole calendar dates. For question-conditioned test examples, evaluation reports combined, transcript-only, and question-only results separately. If paired evidence is present, it also compares combined inputs with and without evidence. These are still teacher-forced 32-token window metrics, not a live turn-taking evaluation.

Question-response links are heuristic, not human-verified. Only explicit `Q.` turns with a question mark or recognized interrogative opening enter strict Q&A training. Short answers that do not fill one target window are omitted. See `docs/data-audit.md` for current counts and known labeling errors.

## External evidence

The Federal Register fetcher supplies dated metadata and abstracts without an API key. Retrieved documents are not joined to questions automatically. To train on evidence, provide JSONL rows keyed by `question_segment_id` as defined in `CONTEXT.md`; training rejects missing, invalid, same-day, or later publication dates because transcript timestamps are unavailable. At generation, use `--as-of` with an optional `--context-query`; evidence is rejected unless the checkpoint's training metadata records evidence pairs. The current local Federal Register sample is unpaired, so current checkpoints do not use it. Evidence input does not guarantee factual grounding, and generated text has no citation head; claim-level accuracy must be evaluated separately.

## Training and generation

```sh
.venv/bin/python -m speechprefix hybrid-train \
  --input data/speeches.jsonl --output checkpoints/speechplan-h.pt --epochs 1
.venv/bin/python -m speechprefix hybrid \
  --model checkpoints/speechplan-h.pt \
  --prefix "We have been discussing our plans" \
  --question "What is the plan?"
```

Use `--max-steps` only for plumbing checks. The old format-1 checkpoint remains loadable for ordinary continuation but predates question conditioning; retrain before using `--question`.

## Limits and prior run

The GovInfo corpus contains mixed presidential documents, not a manually verified speech set. Its current data and heuristic Q&A labels do not support representative quality claims. Model scores are uncalibrated.

The earlier format-1 experiment used a 6.78M-parameter model for five epochs on the then-current corpus and reached test token loss 4.2168 and top-1 accuracy 0.3320. Samples were repetitive and sometimes reproduced editorial notes. Those historical metrics do not evaluate question conditioning or external evidence.

The first format-2 run used the v2 linker for one epoch: 89,443/18,729/15,271 continuation chunks and 2,794/533/437 Q&A chunks across train/validation/test. Test continuation loss/top-1 were 4.9272/0.2704. On those Q&A targets, combined transcript+question scored 4.3369/0.2431, transcript-only 3.9996/0.2574, and question-only 4.8784/0.1869. No evidence pairs were used, so combined and no-evidence results are identical. This run showed no question-conditioning benefit and generated text remained poor. The v3 linker now drops a question association when another named speaker intervenes; its current Q&A window counts are in `docs/data-audit.md`. The format-2 checkpoint is local and git-ignored and must be retrained before comparing against v3.
