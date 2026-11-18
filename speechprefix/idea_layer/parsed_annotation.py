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


def grounded_label(doc):
    """Keep a concrete argument with the predicate that gives it meaning."""
    options = []
    for verb in doc:
        if verb.pos_ != 'VERB' or verb.dep_ not in {'ROOT', 'ccomp', 'xcomp', 'conj'}:
            continue
        for child in verb.children:
            if child.dep_ not in {'dobj', 'obj', 'prep', 'attr'}:
                continue
            if not any(t.pos_ in {'NOUN', 'PROPN'} and not t.is_stop for t in child.subtree):
                continue
            start = min([verb.i, *(t.i for t in verb.children if t.dep_ in {'neg', 'prt'})])
            end = child.right_edge.i + 1
            if child.i <= verb.i or end - start > 16:
                continue
            span = doc[start:end]
            # Avoid stitching together unrelated clauses or losing scope through coordination.
            if any(t.pos_ == 'VERB' and t.i != verb.i for t in span):
                continue
            options.append((child.dep_ != 'prep', verb.dep_ == 'ROOT', end - start, -start, span.text))
    return max(options)[-1] if options else None


def label_span(text, parser, *, frames=False, grounded=False):
    doc = parser(text)
    if frames and (frame := event_label(doc)):
        return frame
    if grounded and (label := grounded_label(doc)):
        return label
    ranked = []
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ not in {'NOUN', 'PROPN'}:
            continue
        tokens = [t for t in chunk if not (t.is_stop or t.is_punct)]
        if not tokens:
            continue
        end = tokens[-1].i + 1
        if grounded:
            for child in head.children:
                if child.dep_ == 'prep' and child.lower_ == 'of' and child.i == chunk.end:
                    if child.right_edge.i - tokens[0].i < 12:
                        end = child.right_edge.i + 1
                        break
        label = doc[tokens[0].i:end].text
        role = {'dobj': 4, 'attr': 4, 'nsubjpass': 4, 'nsubj': 3, 'pobj': 2}.get(head.dep_, 0)
        score = role + min(3, len(tokens)) + int(head.pos_ == 'PROPN') + int(end > chunk.end)
        ranked.append((score, -chunk.start, label))
    return max(ranked)[2] if ranked else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', action='store_true')
    parser.add_argument('--grounded', action='store_true')
    args = parser.parse_args()
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    version = 'classical-grounded-labels-v1' if args.grounded else ('classical-event-frames-v1' if args.frames else VERSION)
    engine = partial(annotate, engine=version, labeler=partial(label_span, parser=nlp, frames=args.frames or args.grounded,
                                                              grounded=args.grounded))
    print(json.dumps(evaluate(args.reference_run, args.output, annotator=engine, engine=version), indent=2))


if __name__ == '__main__':
    main()
