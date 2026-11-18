"""Score reviewed semantic node alignments and conditional typed-link recall."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .transcript_annotations import EDGES, LAYERS, validate_transcript_annotation


def score(candidate, reference, judgments):
    """Null means reviewed and unmatched; an omitted node is unreviewed."""
    mapping = {}
    used = set()
    nodes = {layer: Counter(reference=len(reference[layer]), candidate=len(candidate[layer]),
                            reviewed=0, matched=0) for layer in LAYERS}
    for item in judgments:
        layer, ref, actual = (item[key] for key in ('layer', 'reference_id', 'candidate_id'))
        field = LAYERS[layer]
        if (layer, ref) in mapping or ref not in {n[field] for n in reference[layer]}:
            raise ValueError('duplicate or unknown reference node')
        if actual is not None:
            if (layer, actual) in used or actual not in {n[field] for n in candidate[layer]}:
                raise ValueError('duplicate or unknown candidate node')
            used.add((layer, actual))
        mapping[layer, ref] = actual
        nodes[layer].update(reviewed=1, matched=int(actual is not None))

    links = {}
    for relation in (*EDGES, 'opinion_targets'):
        def endpoints(edge):
            if relation == 'opinion_targets':
                return ('opinions', edge['opinion_id']), (edge['target_type'] + 's', edge['target_id'])
            left, right = EDGES[relation]
            return (left, edge[LAYERS[left]]), (right, edge[LAYERS[right]])

        actual = {endpoints(edge) for edge in candidate[relation]}
        counts = Counter(reference=len(reference[relation]), candidate=len(actual), unreviewed=0,
                         unmatched_endpoint=0, missing_link=0, matched=0)
        for edge in reference[relation]:
            left, right = endpoints(edge)
            if left not in mapping or right not in mapping:
                counts['unreviewed'] += 1
            elif mapping[left] is None or mapping[right] is None:
                counts['unmatched_endpoint'] += 1
            elif ((left[0], mapping[left]), (right[0], mapping[right])) in actual:
                counts['matched'] += 1
            else:
                counts['missing_link'] += 1
        links[relation] = counts
    return {'nodes': nodes, 'links': links}


def evaluate(reference_run, candidate_run, review):
    raw = (reference_run / 'manifest.json').read_bytes()
    if review['reference_manifest_sha256'] != digest(raw):
        raise ValueError('review reference manifest mismatch')
    views = {}
    for record in review['records']:
        if record['view_id'] in views:
            raise ValueError('duplicate review view')
        views[record['view_id']] = record
    records, totals = [], {'nodes': {}, 'links': {}}
    for view, packet, turns, reference in load_references(reference_run, raw):
        data = (candidate_run / f'{view}.json').read_bytes()
        candidate = validate_transcript_annotation(json.loads(data), turns)
        if candidate.get('view_id') != view or candidate['event_id'] != packet['event_id']:
            raise ValueError('candidate identity mismatch')
        record = views.pop(view, None)
        checksum = digest(data)
        if record is not None and record['candidate_sha256'] != checksum:
            raise ValueError('review candidate hash mismatch')
        result = score(candidate, reference, record['judgments'] if record is not None else [])
        records.append(dict(view_id=view, candidate_sha256=checksum, **result))
        for group, metrics in result.items():
            for name, counts in metrics.items():
                totals[group].setdefault(name, Counter()).update(counts)
    if views:
        raise ValueError('review contains unknown view IDs')
    return dict(reference_manifest_sha256=digest(raw), totals=totals, records=records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-run', 'candidate-run', 'review', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    args = cli.parse_args()
    raw = args.review.read_bytes()
    report = evaluate(args.reference_run, args.candidate_run, json.loads(raw))
    write_json(args.output, {**report, 'review_sha256': digest(raw)})
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
