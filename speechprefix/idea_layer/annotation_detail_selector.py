"""Offline parent-conditioned detail retrieval, evaluated with oracle idea labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references, score
from .annotation_candidates import propose
from .classical_annotation import words


def select(bank, turns, occurrence, budget, *, conditioned):
    """Select literal sentence spans, without consulting reference details or quotes."""
    label = set(words(occurrence))
    spans = {(c.turn_id, c.start, c.end) for c in bank if c.kind == 'parsed_sentence'}
    order = {turn: i for i, turn in enumerate(turns)}

    def key(span):
        turn, start, end = span
        tokens = set(words(turns[turn][start:end]))
        relevance = len(tokens & label) / max(1, len(label)) if conditioned else 0
        return -relevance, order[turn], start, end

    return set(sorted(spans, key=key)[:budget])


def evaluate(root, parser, budget):
    raw = (root / 'manifest.json').read_bytes()
    records = []
    totals = {method: Counter() for method in ('parent_conditioned', 'chronological')}
    for view, packet, turns, ref in load_references(root, raw):
        bank = propose(packet, parser)
        mentions = {m['mention_id']: m['evidence'] for m in ref['mentions']}
        points = {p['key_point_id']: set() for p in ref['key_points']}
        for edge in ref['key_point_mentions']:
            points[edge['key_point_id']].update((c['turn_id'], c['exact_quote']) for c in mentions[edge['mention_id']])
        parents = {e['key_point_id']: e['idea_id'] for e in ref['key_point_ideas']}
        counts = {}
        for method in totals:
            predictions = {idea['idea_id']: select(bank, turns, idea['occurrence'], budget,
                                                    conditioned=method == 'parent_conditioned')
                           for idea in ref['ideas']}
            tally = Counter(points=len(points), proposed_links=sum(map(len, predictions.values())))
            for ident, evidence in points.items():
                result = score(predictions[parents[ident]], turns, evidence)
                tally['points_with_exact_evidence'] += bool(result['exact'])
                tally['points_with_overlap_evidence'] += bool(result['overlap_50'])
            counts[method] = tally
            totals[method].update(tally)
        records.append(dict(view_id=view, counts=counts))
    return dict(reference_manifest_sha256=digest(raw), reference_views=len(records), oracle_ideas=True,
                budget_per_idea=budget, parser=parser.meta['version'], totals=totals, records=records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--budget', type=int, default=2)
    args = cli.parse_args()
    if args.budget < 1:
        cli.error('--budget must be positive')
    import spacy
    report = evaluate(args.reference_run, spacy.load('en_core_web_sm'), args.budget)
    write_json(args.output, report)
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
