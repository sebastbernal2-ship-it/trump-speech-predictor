"""Compare offline stance classifiers on identical reference views and spans."""
import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references, score
from .classical_annotation import annotate
from .syntax_stance import classify


def citations(graph):
    mentions = {m['mention_id']: m['evidence'] for m in graph['mentions']}
    stances = {o['opinion_id']: o['stance'] for o in graph['opinions']}
    return {(stances[e['opinion_id']], c['turn_id'], c['exact_quote'])
            for e in graph['opinion_mentions'] for c in mentions[e['mention_id']]}


def compare(expected, emitted, turns):
    counts = Counter(reference=len(expected), candidate=len(emitted), exact=len(expected & emitted))
    for stance in {s for s, _, _ in expected}:
        spans = {(t, m.start(), m.start() + len(q)) for s, t, q in emitted if s == stance
                 for m in re.finditer(f'(?={re.escape(q)})', turns[t])}
        counts['overlap'] += score(spans, turns, {(t, q) for s, t, q in expected if s == stance})['overlap_50']
    return counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    import spacy
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    raw = (args.reference_run / 'manifest.json').read_bytes()
    totals = {method: Counter() for method in ('cue', 'syntax')}
    records = []
    for view, packet, turns, reference in load_references(args.reference_run, raw):
        for method, kwargs in [('cue', {}), ('syntax', {'stance_classifier': partial(classify, parser=parser)})]:
            candidate = annotate({**packet, 'view_id': view}, **kwargs)
            counts = compare(citations(reference), citations(candidate), turns)
            totals[method].update(counts)
            records.append(dict(view_id=view, method=method, counts=counts))
    write_json(args.output, dict(manifest_sha256=digest(raw), parser=parser.meta['version'],
                                 spacy=spacy.__version__, totals=totals, records=records))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
