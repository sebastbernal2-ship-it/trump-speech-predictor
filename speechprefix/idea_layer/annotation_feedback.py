"""Offline, paired six-layer annotation diagnostics and semantic review queue."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .local_annotation_evaluate import compare
from .transcript_annotations import EDGES, LAYERS, validate_transcript_annotation

TEXT = dict(ideas='occurrence', key_points='detail', topics='label', opinions='text', emotions='label')


def anchors(graph):
    result = defaultdict(set)
    for mention in graph['mentions']:
        result['mentions', mention['mention_id']].update(
            (c['turn_id'], c['exact_quote']) for c in mention['evidence'])
    for name, (left, right) in EDGES.items():
        if right == 'mentions':
            for edge in graph[name]:
                result[left, edge[LAYERS[left]]].update(result[right, edge[LAYERS[right]]])
    for edge in graph['topic_ideas']:
        result['topics', edge['topic_id']].update(result['ideas', edge['idea_id']])
    return result


def diagnostics(candidate, reference, limit):
    actual, expected = anchors(candidate), anchors(reference)
    layers = {}
    for layer, field in LAYERS.items():
        stats = Counter(reference=len(reference[layer]), candidate=len(candidate[layer]),
                        no_candidate_on_cited_turn=0, same_turn_only=0, overlapping_quote=0)
        review = []
        for ref in reference[layer]:
            ref_id = ref[field]
            citations = expected[layer, ref_id]
            turns = {t for t, _ in citations}
            ranked = []
            for item in candidate[layer]:
                ident = item[field]
                evidence = actual[layer, ident]
                shared = turns & {t for t, _ in evidence}
                # Substring containment is a retrieval cue, not offset or semantic equivalence.
                contained = any(t == u and (a in b or b in a) for t, a in citations for u, b in evidence)
                if shared:
                    label = TEXT.get(layer)
                    tokens = lambda text: set(re.findall(r'\w+', text.casefold()))
                    a, b = (tokens(ref[label]), tokens(item[label])) if label else (set(), set())
                    ranked.append((contained, len(a & b) / max(1, len(a | b)), ident, item, evidence))
            ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
            stage = ('no_candidate_on_cited_turn' if not ranked else
                     'overlapping_quote' if ranked[0][0] else 'same_turn_only')
            stats[stage] += 1
            review.append(dict(reference=ref, reference_evidence=sorted(citations), diagnostic=stage,
                               candidates=[dict(node=item, evidence=sorted(evidence), verdict=None)
                                           for _, _, _, item, evidence in ranked[:limit]]))
        layers[layer] = dict(counts=stats, review=review)
    return layers


def link_diagnostics(candidate, reference, feedback):
    """Identify whether proposed endpoints exist, then whether a typed edge connects them."""
    proposals = {(layer, row['reference'][LAYERS[layer]]): {
        pair['node'][LAYERS[layer]] for pair in row['candidates']
    } for layer, data in feedback.items() for row in data['review']}
    results = {}
    for relation in (*EDGES, 'opinion_targets'):
        def endpoints(edge):
            if relation == 'opinion_targets':
                return ('opinions', edge['opinion_id']), (edge['target_type'] + 's', edge['target_id'])
            left, right = EDGES[relation]
            return (left, edge[LAYERS[left]]), (right, edge[LAYERS[right]])

        existing = {endpoints(edge) for edge in candidate[relation]}
        counts = Counter(reference=len(reference[relation]), candidate=len(existing),
                         endpoint_not_shortlisted=0, no_edge_between_shortlisted_endpoints=0,
                         edge_between_shortlisted_endpoints=0)
        rows = []
        for edge in reference[relation]:
            left, right = endpoints(edge)
            a, b = proposals[left], proposals[right]
            if not a or not b:
                reason = 'endpoint_not_shortlisted'
            elif any(((left[0], x), (right[0], y)) in existing for x in a for y in b):
                reason = 'edge_between_shortlisted_endpoints'
            else:
                reason = 'no_edge_between_shortlisted_endpoints'
            counts[reason] += 1
            rows.append({'reference_edge': edge, 'diagnostic': reason})
        results[relation] = {'counts': counts, 'review': rows}
    return results


def build(reference_run, runs, limit):
    raw = (reference_run / 'manifest.json').read_bytes()
    reports = {name: dict(exact=defaultdict(Counter), anchors=defaultdict(Counter), links=defaultdict(Counter), records=[]) for name in runs}
    for view, packet, turns, reference in load_references(reference_run, raw):
        for name, path in runs.items():
            data = (path / f'{view}.json').read_bytes()
            candidate = validate_transcript_annotation(json.loads(data), turns)
            if candidate.get('view_id') != view or candidate['event_id'] != packet['event_id']:
                raise ValueError(f'{name}: candidate identity mismatch')
            exact = compare(candidate, reference)
            feedback = diagnostics(candidate, reference, limit)
            report = reports[name]
            for key, counts in exact.items():
                report['exact'][key].update(counts)
            for key, value in feedback.items():
                report['anchors'][key].update(value['counts'])
            links = link_diagnostics(candidate, reference, feedback)
            for key, value in links.items():
                report['links'][key].update(value['counts'])
            report['records'].append(dict(view_id=view, event_id=packet['event_id'],
                                          candidate_sha256=digest(data), exact=exact, layers=feedback, links=links))
    return dict(reference_manifest_sha256=digest(raw), top_k=limit, runs=reports)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--candidate-run', action='append', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--top-k', type=int, default=3)
    args = cli.parse_args()
    runs = {p.name: p for p in args.candidate_run}
    if args.top_k < 1 or len(runs) != len(args.candidate_run):
        cli.error('use a positive top-k and distinct run names')
    report = build(args.reference_run, runs, args.top_k)
    write_json(args.output, report)
    print(json.dumps({name: run['anchors'] for name, run in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
