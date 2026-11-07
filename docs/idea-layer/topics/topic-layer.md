# Topic layer above speech conversation points

**Status:** diagnostic topic-grouping prototype (`speech-idea-topic-grouping-v1.1`); not semantic gold and not an active predictor input.

## Layer contract

The useful hierarchy is:

1. **Transcript mentions/evidence** — exact speaker-grounded text, preserving what was said.
2. **Conversation points** — specific propositions, claims, positions, proposals, or appeals. These are the leaf units; keep distinct claims and story references separate.
3. **Topics** — broader issue/thread labels that organize one or more points for interpretation and navigation.

A topic is not a replacement summary for its members. A topic can group distinct claims, policy proposals, and events that share a meaningful subject, but membership does not assert that the points are paraphrases, the same story, causally related, or interchangeable forecast targets. The points keep their original IDs, summaries, evidence, and boundaries. Use a separate typed relation if a stronger relation (same story, supports, specific-to, etc.) is warranted; do not infer it from topic membership.

The prototype assigns each point exactly one primary topic (`topic_memberships[].relation = organized_under`). If later evidence justifies multi-topic membership, version the contract and validate it explicitly instead of silently duplicating points. The intended use is to evaluate topic-level and point-level prediction as separate targets. The current topic layer is diagnostic and is not integrated into the active predictor. Any features for either task must be derived from eligible pre-event sources; target-speech mentions, points, topics, and derived labels remain outcome-only. See [`multilevel-prediction-architecture.md`](../prediction/multilevel-prediction-architecture.md).

## Implementation

`research/idea_layers/topics/group_speech_ideas_into_topics.py` creates a separate, provenance-bound topic-layer artifact from a layered transcript outcome JSONL. It requires complete one-to-one point coverage, unique topic IDs, and no unknown or duplicate memberships. Validation is structural only; it cannot establish topical coherence. Source ideas are not mutated.

Run example:

```bash
uv run python research/idea_layers/topics/group_speech_ideas_into_topics.py \
  --input data/source-archives/speech-content/outcomes/<transcript-outcomes>.jsonl \
  --output data/source-archives/speech-content/topics/<topic-layer>.jsonl
```

Outputs and manifests belong under Git-ignored `data/source-archives/`; do not stage or commit them.

## Lexington-like short-speech demonstration: packet-01

The v2.6 transcript-only point extractor produced 10 candidate points for the short 2025 public-safety remarks. Topic grouping v1.1 organized them into three topic groups while retaining all 10 point IDs:

- **Response to the Guard shooting:** condemnation of the attack; support/prayers for the wounded; promised punishment/justice.
- **Immigration and national security:** suspect's reported Afghan entry/status; broad claims about unvetted admissions; Afghan-entry review; removal proposals.
- **Capital security mission:** claimed D.C. task-force crime reductions; additional 500 troops/continuing the mission; thanks to service/security personnel.

The current local output is `data/source-archives/keypoint-reference-sample-v2/packet-01-topic-layer-v1.1.jsonl`, output SHA-256 `f0385c2945297c315d3d22e09a856c2bba6f89f2d8cad30c1349f59a780884f7`; its manifest SHA-256 is `2734f8dd121b129c89cb4e80794c5bba0b7db8c0cb7ff001f7bc219d20430288`.

**Important point-boundary issue:** the extractor did capture the speech's Minnesota/Somali passages as transcript mentions, but linked them under the broad `unvetted_entry_threat` point. That may be an under-split: allegations about Somali communities/representatives are independently assessable from the general claim about 20 million unvetted entrants and should be considered as their own point before topic grouping. The topic layer cannot repair an omitted or over-broad point; it only organizes the point inventory it receives. If separated, those points may still sit under a broader immigration/national-security topic without being merged.

This one-speech, same-provider run demonstrates the representation, not quality. Human review remains necessary for topic coherence, topic breadth, and the underlying point boundaries.
