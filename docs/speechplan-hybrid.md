# SpeechPlan-H

SpeechPlan-H is a new, from-scratch continuation model in this repository. It does not load GPT-2 weights, Stanford NCM weights, or any downloaded model checkpoint. It uses the installed Transformers GPT-2 implementation only as a randomly initialized autoregressive decoder. Its architecture combines:

1. A context encoder that summarizes the observed prefix.
2. A training-only future encoder and residual vector quantizer. It learns discrete codebooks at 1, 4, 16, and 32 positions over the next 32-token segment.
3. A coarse-to-fine planner that predicts each code level from the prefix and its preceding predicted level.
4. A GPT-style token decoder with cross-attention to all planned code embeddings. It also reads the observed prefix through causal self-attention.

Training jointly minimizes next-token cross-entropy, hierarchical code-prediction cross-entropy, and vector-quantizer reconstruction/commitment loss. During inference, the future encoder is absent: the planner predicts codes from the prefix, and the decoder generates the next segment conditioned on those codes.

This is an original prototype design here, not a claim of globally novel research or equivalent scale/quality to a pretrained LLM. It starts randomly initialized and learns a word-level vocabulary from the local training split. The GovInfo corpus is mixed presidential material, not a manually verified speech set; a chronological holdout on this corpus is only a plumbing check. Do not interpret its scores as representative speech quality.

The initial 5-epoch local run used 89,443/18,729/15,271 train/validation/test chunks and selected the checkpoint by validation loss. The 6.78M-parameter model reached test token loss 4.2168 and top-1 accuracy 0.3320 on this split. Qualitative samples remain repetitive and sometimes reproduce editorial notes, so this is evidence that training/inference works, not that the model is useful. The checkpoint is local and git-ignored at `checkpoints/speechplan-h.pt`.

Train locally, with no model download:

```sh
.venv/bin/python -m speechprefix hybrid-train \
  --input data/speeches.jsonl --output checkpoints/speechplan-h.pt --epochs 1
.venv/bin/python -m speechprefix hybrid \
  --model checkpoints/speechplan-h.pt --prefix "Ladies and gentlemen, thank you"
```

`hybrid-train --max-steps 2` caps optimizer steps for a smoke run; it still computes validation and test metrics, which are not meaningful from such a partial run. The trainer splits whole calendar dates chronologically and selects the checkpoint by validation loss. Pre-speech forecasting remains a separate task and is not an input to this continuation model.
