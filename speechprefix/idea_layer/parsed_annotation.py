"""Offline syntactic subject labels for the classical annotation graph."""
import argparse
from functools import partial
import json
from pathlib import Path

from .classical_annotation import annotate
from .local_annotation_evaluate import evaluate

VERSION = 'classical-parsed-labels-v1'


def label_span(text, parser):
    doc = parser(text)
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
    args = parser.parse_args()
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    engine = partial(annotate, engine=VERSION, labeler=partial(label_span, parser=nlp))
    print(json.dumps(evaluate(args.reference_run, args.output, annotator=engine, engine=VERSION), indent=2))


if __name__ == '__main__':
    main()
