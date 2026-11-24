# STATE

Project: trump-speech-predictor local annotation distillation.
Branch feat/distilled-annotator, worktree ~/code/tsp-distill (base 3a61895 after history rewrite; backups ~/code/tsp-master-backup*.bundle).

## Done
- distill_data / train / eval / calibrate / matcher / graph_eval scripts.
- 67 frozen Codex drafts are the only labels.
- 5-fold LoRA Qwen3-0.6B results: ideas R72/P79 (thr .50), key pts R39/P60, topics R41/P45, opinions R47/P55, tone R50/P59.
- Quotes 90.8% exact; 22% of views have broken links; matched key points attach to the right parent only 57%.
- 375 AI-judge labels (same 181 / diff 154 / broader 28 / narrower 12) at distill-v1/judge-a-*.json.
- Cosine matcher best F1 .75 strict / .80 multi.
- Fine-tuned distilroberta cross-encoder collapsed.

## Data
- Dir: /home/vorte/code/trump-speech-predictor/data/source-archives/speech-content/outcomes/distill-v1 (ignored local data; never stage).

## In flight
- Nothing running.
- Qwen3-4B 5-fold run NOT done (q4b/ missing; download of shards 1-2 incomplete).
- NLI matcher (cross-encoder/nli-deberta-v3-small) NOT run.
- Codex sandbox cannot see GPU or write the data dir: GPU jobs must run via a worker with a real shell, not codex_exec.

## Teacher labels
- Codex 300-view sample at outcomes/teacher-sample-v1; only 5 drafts valid (80% rejected for inexact quotes); labeling paused.
- Codex accounts live: sean; never call codex-account use.

## Next 3
1. Finish Qwen3-4B download, run folds 0-4 (distill_train --model Qwen/Qwen3-4B --epochs 4 --output $B/q4b/foldF), evaluate with distill_eval --layer-thresholds and distill_graph_eval. Check nvidia-smi first (weight-drift-reader project used the GPU).
2. NLI matcher via distill_matcher.py.
3. LLM-judge scorer on fixed held-out set; user spot-checks ~40 judge labels; then hierarchy task and more teacher labels.

## Traps
- Commit as sebastbernal2-ship-it with Claude co-author trailer.
- Single branch, no PR until goal met; AGENTS.md delivery workflow applies at the end.
- Weekly Claude usage limit hit once (reset Nov 23 7pm PT).

## Files
- speechprefix/idea_layer/distill_*.py
- tests/idea_layers/outcomes/test_distill_*.py
