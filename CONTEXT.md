# Speech forecasting

This context separates continuation of an observed transcript from forecasting content before an event. It also distinguishes speaker-attributed turns from plain transcript paragraphs.

## Language

**Continuation prefix**:
Text already observed from the current speech before the target segment.
_Avoid_: Prompt, history when referring only to this speech

**Transcript segment**:
A contiguous part of one source transcript, split at a paragraph or explicit turn boundary and retaining its source and position.
_Avoid_: Chunk without source position

**Speaker turn**:
A transcript segment with an explicit speaker label in the source. A question mark alone does not identify a questioner.
_Avoid_: Inferred speaker

**Question-response pair**:
A `Q.`-labeled turn with a question mark or recognized interrogative opening, linked to one or more subsequent president-labeled paragraphs until another explicit speaker intervenes. The link is heuristic and is not human-verified.
_Avoid_: Answer label inferred only from punctuation or topic

**Reporter prompt**:
A `Q.`-labeled turn without a question mark or recognized interrogative opening. It may be a floor cue, comment, or question fragment, so it is excluded from strict Q&A training.
_Avoid_: Question

**Event-time evidence**:
Information published or otherwise available no later than the prediction timestamp, with a recorded source and publication time.
_Avoid_: World context without an as-of date

**Idea state**:
A model-internal summary of ideas in earlier observed segments. It is not a verified human topic annotation.
_Avoid_: Concept label unless human-labeled

## Dated evidence-pair JSONL

Each row maps one question segment to source records published strictly before that speech date. Because the transcript corpus stores dates but not question timestamps, same-day evidence is rejected:

```json
{"question_segment_id":"DCPD-...:0001","documents":[{"context_id":"...","published_at":"2025-01-20","title":"...","abstract":"..."}]}
```

Training requires a source identifier or URL and rejects missing, invalid, same-day, or later publication dates. Do not attach unpaired context records by guesswork.
