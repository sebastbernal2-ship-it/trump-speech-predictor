# Speech forecasting

This project separates pre-speech forecasting, ordinary transcript continuation, and question-anchored prediction of a target speaker's next response. Time-coded turns preserve who spoke and what was observable at each prediction point.

## Language

**Continuation prefix**:
Text already observed from the current event before a target response.
_Avoid_: Prompt when referring only to the transcript

**Transcript span**:
A source-delimited interval with a time range, speaker label, and transcript text. A source may split one continuous speech turn into several spans.
_Avoid_: Paragraph segment when timestamps and speaker turns are available

**Question speaker label**:
A source speaker category labeled `Question`. It identifies the questioner channel, but the span may be a prompt, fragment, or clarification rather than a complete question.
_Avoid_: Inferred question-response relation

**Prediction point**:
The time through which transcript and external context are known before predicting the target speaker's next span.
_Avoid_: Target start time when the preceding question or clarification is still incomplete

**Prediction trigger**:
One or more observed spans selected by a reviewer as the focus for predicting the target speaker's next response. A trigger can be a reporter's question or follow-up, or the target speaker's own clarifying question.
_Avoid_: Automatically selected adjacent span

**Response alignment**:
A reviewed link between visible trigger spans and the target speaker's following span or response status. Adjacency alone does not establish that a response answers a question.
_Avoid_: Automatically paired question and next paragraph

**Idea label**:
A human-reviewed semantic category for a response, evaluated separately from its transcript wording.
_Avoid_: Model-internal representation

**Idea state**:
A model-internal summary of ideas expressed in earlier observed spans. It is not a verified human annotation.
_Avoid_: Idea label

**Speech theme**:
A broad message or idea that may span several parts of one speech.
_Avoid_: Plan code

**Speech section**:
A contiguous part of a speech that develops a theme or subtopic.
_Avoid_: Talking point

**Talking point**:
A specific claim, story, example, promise, attack, or mention developed within a section and grounded in transcript spans.
_Avoid_: Phrase, plan code

**Speech plan**:
A human-readable, ordered hierarchy of themes, sections, and talking points, kept distinct from the exact transcript wording.
_Avoid_: Model-internal representation

**Plan code**:
A model-internal quantized representation with no verified human-readable meaning.
_Avoid_: Speech theme, talking point, concept label

**Event-time evidence**:
Information with a recorded source and publication time earlier than the prediction point. If only dates are available, same-day evidence is excluded.
_Avoid_: World context without an as-of time
