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
An explicit question turn and a subsequent president-labeled turn linked in transcript order. This link is provisional unless reviewed.
_Avoid_: Answer label inferred only from punctuation or topic

**Event-time evidence**:
Information published or otherwise available no later than the prediction timestamp, with a recorded source and publication time.
_Avoid_: World context without an as-of date

**Idea state**:
A model-internal summary of ideas in earlier observed segments. It is not a verified human topic annotation.
_Avoid_: Concept label unless human-labeled
