# Ideas stage

**Output unit:** specific talking point/idea. This is finer than a topic and coarser than sentences, words, or characters. The active classifier still predicts the existing compact issue-phrase IDs; they are weak lexical labels, not yet an independently adjudicated point ontology.

**Status:** active package. The predictor and chronological evaluator live in `model.py`; the `context-idea-classifier-evaluate` and `context-idea-predict` CLI commands call this module.

This stage owns direct prediction of target-speech ideas from preceding-week context. Its public prediction output is a ranked list of idea IDs and uncalibrated scores bound to the hash of the fitted evaluation/model artifact. Training and evaluation remain inside this stage. The active model does not consume topic predictions; topic-conditioned ideas are a separate future ablation. Do not add sentence realization, word/character generation, or speech-history recurrence features here. Keep the stage's inference contract and leakage tests stable as the model evolves.

Transcript-derived ideas are outcome labels for their own event, never same-event features. Future sentence, word, and character stages must consume frozen prediction artifacts rather than reaching into this stage's mutable internals.
