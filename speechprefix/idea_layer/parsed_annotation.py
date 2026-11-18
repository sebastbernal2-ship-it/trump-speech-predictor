"""Offline syntactic subject labels for the classical annotation graph."""
import argparse
from functools import partial
import json
from pathlib import Path

from .classical_annotation import annotate
from .local_annotation_evaluate import evaluate

VERSION = 'classical-parsed-labels-v1'


# Keep attribution and polarity in the evidence/stance layers, not in the subject label.
NOMINAL = {'run': 'management', 'investigate': 'investigation', 'prosecute': 'prosecution',
           'fund': 'funding', 'regulate': 'regulation', 'ban': 'ban', 'endorse': 'endorsement'}


def event_label(doc):
    for verb in doc:
        action = NOMINAL.get(verb.lemma_)
        if verb.pos_ != 'VERB' or not action or verb.dep_ not in {'ROOT', 'ccomp'}:
            continue
        subject = next((t for t in verb.children if t.dep_ == 'nsubj'), None)
        obj = next((t for t in verb.children if t.dep_ in {'dobj', 'obj'}), None)
        if subject is None or obj is None or subject.pos_ not in {'NOUN', 'PROPN'}:
            continue
        phrases = {chunk.root.i: chunk.text for chunk in doc.noun_chunks}
        if subject.i in phrases and obj.i in phrases:
            return f'{action} of {phrases[obj.i]} by {phrases[subject.i]}'
    return None


def label_span(text, parser, *, frames=False):
    doc = parser(text)
    if frames and (frame := event_label(doc)):
        return frame
    ranked = []
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ not in {'NOUN', 'PROPN'}:
            continue
        tokens = [t for t in chunk if not (t.is_stop or t.is_punct)]
        if not tokens:
            continue
        label = doc[tokens[0].i:tokens[-1].i + 1].text
        role = {'dobj': 4, 'attr': 4, 'nsubjpass': 4, 'nsubj': 3, 'pobj': 2}.get(head.dep_, 0)
        score = role + min(3, len(tokens)) + int(head.pos_ == 'PROPN')
        ranked.append((score, -chunk.start, label))
    return max(ranked)[2] if ranked else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', action='store_true')
    args = parser.parse_args()
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    version = 'classical-event-frames-v1' if args.frames else VERSION
    engine = partial(annotate, engine=version, labeler=partial(label_span, parser=nlp, frames=args.frames))
    print(json.dumps(evaluate(args.reference_run, args.output, annotator=engine, engine=version), indent=2))


if __name__ == '__main__':
    main()
