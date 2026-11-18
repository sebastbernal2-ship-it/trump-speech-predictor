"""Open-vocabulary extractive annotation with replaceable span similarity."""
from collections import Counter
import re

from .spoken_evidence import has_spoken_evidence
from .transcript_annotations import EDGES, LAYERS, validate_transcript_annotation

VERSION = 'classical-extractive-v1'
WORD = re.compile(r"[\w]+(?:['’-][\w]+)*")
STOP = set('the a an and or but it this that we they i you he she our their to of for in on at by with from is are was were be have has about as not do did will would can know think very just really now going'.split())
STANCE = tuple((label, re.compile(pattern, re.I)) for label, pattern in (
    ('opposes', r"\b(?:oppose|reject|should not|do not support|don't support)\b"),
    ('supports', r'\b(?:support|endorse|favor)\b'),
    ('proposes', r'\b(?:propose|we should|we must)\b'),
    ('requests', r'\b(?:urge|ask|call on)\b'),
    ('forecasts', r'\b(?:expect|predict|will)\b'),
    ('asserts', r'\b(?:is|are|was|were|has|have|believe|think)\b'),
))
TONE = tuple((label, re.compile(pattern, re.I)) for label, pattern in (
    ('praise', r'\b(?:excellent|wonderful|outstanding|great job)\b'),
    ('condemnation', r'\b(?:disgrace|disgraceful|outrageous|shameful)\b'),
    ('concern', r'\b(?:concerned|worried|alarming)\b'),
    ('gratitude', r'\b(?:grateful|thank you|appreciate)\b'),
))


def words(text):
    return [w.casefold() for w in WORD.findall(text) if w.casefold() not in STOP and len(w) > 1]


def lexical_similarity(labels):
    tokens = [set(words(label)) for label in labels]
    return lambda i, j: len(tokens[i] & tokens[j]) / max(1, len(tokens[i] | tokens[j]))


def cue_stance(span):
    if '?' not in span:
        return next((name for name, pattern in STANCE if pattern.search(span)), None)
    return None


def annotate(packet, *, similarity=lexical_similarity, threshold=.45, engine=VERSION,
             labeler=None, stance_classifier=cue_stance):
    row = dict(artifact_type='transcript-annotation', schema_version=1, coverage='full_pass',
               event_id=packet['event_id'], view_id=packet['view_id'], generator={'engine': engine},
               annotation_pass='local_engine_candidate', **{k: [] for k in (*LAYERS, *EDGES, 'opinion_targets')})
    turns = {t['turn_id']: t['text'] for t in packet['turns'] if t['role'] == 'target' and has_spoken_evidence(t['text'])}
    spans = [(turn, match[0].strip()) for turn, text in turns.items()
             for match in re.finditer(r'[^.!?\n]+[.!?]*', text) if has_spoken_evidence(match[0]) and len(words(match[0])) >= 2]
    frequency = Counter(w for _, span in spans for w in set(words(span)))
    labels = []
    for _, span in spans:
        tokens = words(span)
        names = re.findall(r'\b(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+\b', span)
        pairs = list(zip(tokens, tokens[1:]))
        fallback = names[0] if names else ' '.join(max(pairs, key=lambda pair: sum(frequency[w] for w in pair)))
        labels.append((labeler(span) if labeler else None) or fallback)
    similar = similarity(labels)

    def node(layer, **fields):
        key = LAYERS[layer]
        ident = f'{key[:-3]}_{len(row[layer]) + 1}'
        row[layer].append({key: ident, **fields})
        return ident

    def link(name, a, b):
        left, right = EDGES[name]
        row[name].append({LAYERS[left]: a, LAYERS[right]: b})

    ideas, topics = [], []
    for i, ((turn, span), label) in enumerate(zip(spans, labels)):
        mention = node('mentions', evidence=[{'turn_id': turn, 'exact_quote': span}])
        found = next((ident for j, ident in ideas if similar(i, j) >= threshold), None)
        if found is None:
            found = node('ideas', occurrence=label, uncertainty='Extractive subject candidate; review scope.')
            ideas.append((i, found))
            topic = next((ident for j, ident in topics if similar(i, j) >= threshold / 2), None)
            if topic is None:
                topic = node('topics', label=f'{label} discussion', uncertainty='Provisional grouping label.')
                topics.append((i, topic))
            link('topic_ideas', topic, found)
        link('idea_mentions', found, mention)
        detail = re.search(r'\b(?:including|such as|for example|specifically)\s+([^.;!?]+)', span, re.I)
        point = None
        if detail and detail[1].strip():
            point = node('key_points', detail=detail[1].strip(), uncertainty='Detail cue; review attachment.')
            link('key_point_ideas', point, found)
            link('key_point_mentions', point, mention)
        if stance := stance_classifier(span):
            opinion = node('opinions', text=span, attribution='speaker', stance=stance,
                           uncertainty='Surface cue; attribution, scope and intent require review.')
            link('opinion_mentions', opinion, mention)
            row['opinion_targets'].append(dict(opinion_id=opinion, target_type='idea', target_id=found))
        for tone, pattern in TONE:
            if pattern.search(span):
                emotion = node('emotions', label=tone, uncertainty='Textual cue only; not an emotional state.')
                link('emotion_mentions', emotion, mention)
    if not row['ideas']:
        row['empty_reason'] = 'No extractable span; baseline abstention.'
    return validate_transcript_annotation(row, turns)
