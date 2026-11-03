# Explicit hierarchical speech planner

## The idea in model terms

The speech plan has three human-readable levels:

1. **Theme:** a broad message that can span several parts of the speech.
2. **Section:** a contiguous passage developing a theme or subtopic.
3. **Talking point:** a specific claim, story, example, promise, attack, or mention within a section.

The exact phrases and words are the **surface wording**, generated after the semantic plan. These levels are distinct from SpeechPlan-H's existing VQ plan codes, which remain opaque learned representations.

## Intended in-speech path

```mermaid
flowchart TD
    PREFIX["Observed transcript prefix"] --> STATE["Track observed themes, sections, and points"]
    STATE --> THEME["Predict next theme or whether to continue current theme"]
    THEME --> SECTION["Predict next section or subtopic"]
    SECTION --> POINT["Predict next talking point"]
    POINT --> PLAN["Encode explicit plan as conditioning memory"]
    PREFIX --> LOCAL["Optional local 32-token VQ planner"]
    PLAN --> WORDS["Autoregressive word generator"]
    LOCAL --> WORDS
    PREFIX --> WORDS
    WORDS --> TEXT["Generate the next text segment"]
    TEXT --> NEXT["Observe actual next transcript spans"]
    NEXT --> STATE
```

The hierarchy gives two separate views of the same speech. **Before the speech**, the pre-speech task may predict the whole event-level outline from prior speeches and genuinely available event metadata. **During the speech**, the continuation task predicts only the next theme/section/point from the observed prefix and previously predicted plan nodes, then updates state after new transcript spans arrive. It must not read later nodes from the full gold outline. The word generator verbalizes the predicted node; it does not get to use future target text. Keep these evaluations separate.

The existing VQ planner can remain as a fine-grained wording aid. Its 1/4/16/32 codes must not be renamed as themes or ideas. The explicit hierarchy needs its own training targets and evaluation.

## Training data contract

A reviewed plan is stored separately from the source event and points back to target-speaker transcript turns. Each node has a short summary and supporting `turn_ids`; parent nodes include the turns covered by their children. Sibling points may share a turn when one source span contains multiple ideas. The event transcript remains the source of exact wording.

```json
{
  "schema_version": 1,
  "event_id": "sample-event",
  "review_status": "draft",
  "reviewed_by": null,
  "themes": [
    {
      "id": "theme-1",
      "summary": "Example broad theme",
      "turn_ids": ["sample-event:turn:0000", "sample-event:turn:0001"],
      "sections": [
        {
          "id": "section-1",
          "summary": "Example subtopic",
          "turn_ids": ["sample-event:turn:0000", "sample-event:turn:0001"],
          "talking_points": [
            {
              "id": "point-1",
              "summary": "Example claim or mention",
              "turn_ids": ["sample-event:turn:0000"]
            }
          ]
        }
      ]
    }
  ]
}
```

Only `review_status: "reviewed"` plans with a reviewer and valid target-speaker turn references are eligible for supervised training. Draft or machine-suggested plans are not gold labels. Annotators should ground every node in the cited transcript turns, avoid inferring intent not stated in the transcript, and allow overlapping turn references when a source span contains multiple points. The local 20-event review batch is only a draft, not training data. Review a small pilot and check annotator agreement before scaling; do not train an explicit semantic hierarchy on automatically inferred adjacent spans or treat VQ codes as semantic labels.

## Evaluation boundaries

Evaluate theme, section, and talking-point prediction separately from exact wording. Split by whole event dates. For in-speech predictions, compare only plan nodes grounded in transcript spans that occur after the prediction point; do not expose those nodes or their text in the input. Keep this task separate from pre-speech forecasts and question-anchored response prediction.

## Explicit path model and label policies

`explicit-plan-train` trains a separate, from-scratch causal model that predicts one theme → section → talking-point path from an observed prefix. It supports two deliberately separate label policies:

- `reviewed` accepts only human-reviewed plans and is the route to semantic-plan evaluation.
- `weak-extractive` accepts only plans marked as drafts by the named labeler. It uses the highest-score Factba.se/IPTC source topic as the theme when present (otherwise whole-event keywords), fixed 15-target-turn keyword windows for sections, and fixed 3-turn keyword windows for talking points. IDF weights are computed only from the chronological training-date partition. These are real-transcript/source-metadata pseudo-labels, not human review or verified ideas; their scores measure reproduction of the labeling heuristic.

Both policies split by whole event dates and exclude the target point's supporting turns from its transcript input. Test metrics include token loss, teacher-forced token accuracy, and greedy exact-path match. Exact text match is not semantic idea accuracy. The model predicts the path only; it does not generate surface wording or condition SpeechPlan-H's decoder. Every run records its label policy, data hashes, and metrics; see `docs/model-versioning.md`.
