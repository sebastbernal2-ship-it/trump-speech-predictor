# Speech forecasting

## Primary project direction: the standalone idea layer

The first goal is event-specific ideas, not wording: rank what Trump may address before a scheduled speech using current, timestamped news, his attributable public posts, and event details known by the cutoff; then append updates using only speech turns already observed. This layer must stay independent of continuation, speech generation, and Q&A response models. Its package boundary is `speechprefix/idea_layer/`; see `docs/idea-layer/README.md`.

Use an exact prediction cutoff for every forecast. Evidence published or first observed at/after that cutoff is ineligible. Every candidate keeps source citations. A transcript outcome may be used only after the event to evaluate the frozen forecast, never to create its inputs.

The project keeps three concepts distinct: a forecast idea is a model candidate; an outcome annotation is a source-attributed transcript interpretation (which may be assistant-generated and weak); semantic gold requires an explicitly stronger review process. Never describe AI-generated outcome annotations as human-reviewed or use them to claim semantic accuracy.

Continuation, full-speech planning, and question-anchored response prediction remain separate research tasks, not part of the first idea-layer deliverable. Time-coded turns preserve what was observable at each live-update point.

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
