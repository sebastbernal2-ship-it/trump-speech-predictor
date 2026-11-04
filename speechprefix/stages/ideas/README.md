# Ideas stage

**Status:** active package. The predictor and chronological evaluator live in `model.py`; the `context-idea-classifier-evaluate` and `context-idea-predict` CLI commands call this module.

This stage owns prediction of target-speech ideas from preceding-week context. Its public prediction output is a ranked list of idea IDs and uncalibrated scores bound to the hash of the fitted evaluation/model artifact. Training and evaluation remain inside this stage. Do not add sentence realization or speech-history recurrence features here. Keep the stage's inference contract and leakage tests stable as the model evolves.
