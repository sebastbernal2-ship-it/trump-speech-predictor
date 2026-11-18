"""Queue cross-scale idea comparisons for evidence-grounded review."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .idea_agreement import subject_signature
from .idea_pair_review import evidence_by_idea


def possible_promotions(inventory, reference, turns):
    """Retrieve overlapping key-point evidence; do not infer hierarchy from overlap."""
    mentions = {m['mention_id']: m['evidence'] for m in reference['mentions']}
    parents = {p['key_point_id']: p['idea_id'] for p in reference['key_point_ideas']}
    ideas = {i['idea_id']: i['occurrence'] for i in reference['ideas']}
    idea_evidence = evidence_by_idea(reference)
    detail_evidence = {p['key_point_id']: [] for p in reference['key_points']}
    for edge in reference['key_point_mentions']:
        detail_evidence[edge['key_point_id']].extend(mentions[edge['mention_id']])
    for index, idea in enumerate(inventory):
        spans = idea['evidence']
        for span in spans:
            turn, a, b = span['turn_id'], span['start'], span['end']
            if turn not in turns or not 0 <= a < b <= len(turns[turn]) or turns[turn][a:b] != span['quote']:
                raise ValueError('invalid target-speaker evidence span')
        def shares(citations):
            return any(s['turn_id'] == c['turn_id'] and
                       (s['quote'] in c['exact_quote'] or c['exact_quote'] in s['quote'])
                       for s in spans for c in citations)

        top_level_options = [ident for ident in ideas if shares(idea_evidence[ident])]
        for point in reference['key_points']:
            ident = point['key_point_id']
            if not shares(detail_evidence[ident]):
                continue
            parent = parents[ident]
            yield dict(inventory_index=index, candidate=idea, reference_key_point=point,
                       reference_detail_evidence=detail_evidence[ident], reference_parent_id=parent,
                       reference_parent=ideas[parent], overlapping_reference_idea_ids=top_level_options,
                       exact_detail_label=subject_signature(idea['label']) == subject_signature(point['detail']),
                       subject_validity=None, scale_relation_to_parent=None,
                       attachment_verdict=None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-run', 'inventory-report', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    cli.add_argument('--method', default='sparse')
    args = cli.parse_args()
    raw = (args.reference_run / 'manifest.json').read_bytes()
    inventory_raw = args.inventory_report.read_bytes()
    report = json.loads(inventory_raw)
    if report['reference_manifest_sha256'] != digest(raw):
        raise ValueError('inventory reference manifest mismatch')
    views = dict(report['inventories'][args.method])
    records = []
    counts = Counter()
    for view, _, turns, ref in load_references(args.reference_run, raw):
        inventory = views.pop(view, [])
        counts.update(reference_views=1, candidate_ideas=len(inventory), reference_key_points=len(ref['key_points']))
        records.extend(dict(view_id=view, **row) for row in possible_promotions(inventory, ref, turns))
    if views:
        raise ValueError('inventory contains unknown views')
    counts.update(review_pairs=len(records),
                  candidate_ideas_queued=len({(r['view_id'], r['inventory_index']) for r in records}),
                  exact_detail_label_pairs=sum(r['exact_detail_label'] for r in records))
    write_json(args.output, dict(reference_manifest_sha256=digest(raw), inventory_sha256=digest(inventory_raw),
                                 method=args.method, counts=counts, records=records,
                                 verdicts=dict(subject_validity=['valid', 'unsupported', 'not_substantive', 'unclear'],
                                               scale_relation_to_parent=['same', 'narrower', 'broader', 'different', 'unclear'],
                                               attachment_verdict=['warranted', 'not_warranted', 'unclear'])))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
