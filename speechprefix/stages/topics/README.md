# Topics stage (reserved)

**Status:** scaffold only; no predictor, training set, or evaluation task is implemented.

This stage predicts broad issue/theme units from strictly pre-event context. A topic is broader than an idea/talking point; it does not identify a real-world story and must not absorb point-level claims. Topic IDs/definitions need independently reviewed transcript-outcome references before supervised evaluation.

Possible interfaces:

- Input: a versioned, as-of feature bundle or eligible pre-event context snapshot.
- Output: ranked topic IDs/scores in an immutable artifact with input/model hashes and cutoff metadata.
- Optional downstream input: a frozen **predicted** topic artifact for the ideas stage. The current active ideas model does not consume topic predictions.

Keep topic-level coverage/ranking and calibration separate from idea-level semantic matching. Compare direct topic prediction, point-prediction aggregation, and topic-conditioned idea prediction as separately versioned chronological experiments.