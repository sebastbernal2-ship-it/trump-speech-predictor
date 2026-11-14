"""Compare a local extraction run with frozen development reference graphs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .local_annotation_engine import VERSION, annotate
from .transcript_annotations import EDGES, LAYERS, validate_transcript_annotation


def normalized(text):
    return ' '.join(text.casefold().split())


def features(row):
    """Compare labels and edges independent of serialization order and node IDs."""
    nodes = {layer: {} for layer in LAYERS}
    for layer, id_field in LAYERS.items():
        for node in row[layer]:
            if layer == 'mentions':
                value = tuple(sorted((e['turn_id'], e['exact_quote']) for e in node['evidence']))
            elif layer == 'opinions':
                value = node['stance'], node['attribution'], normalized(node['text'])
            else:
                value = normalized(node[{'ideas': 'occurrence', 'key_points': 'detail',
                                         'topics': 'label', 'emotions': 'label'}[layer]])
            nodes[layer][node[id_field]] = value
    groups = {layer: Counter(mapping.values()) for layer, mapping in nodes.items()}
    groups.update({name: Counter((nodes[left][edge[LAYERS[left]]], nodes[right][edge[LAYERS[right]]])
                                 for edge in row[name]) for name, (left, right) in EDGES.items()})
    groups['opinion_targets'] = Counter(
        (nodes['opinions'][edge['opinion_id']], edge['target_type'],
         nodes[edge['target_type'] + 's'][edge['target_id']]) for edge in row['opinion_targets'])
    return groups


def compare(candidate, reference):
    actual, expected = features(candidate), features(reference)
    return {name: {'matched': sum((actual[name] & value).values()),
                   'candidate': sum(actual[name].values()), 'reference': sum(value.values())}
            for name, value in expected.items()}


def evaluate(reference_run: Path, output: Path):
    manifest_bytes = (reference_run / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    refs = {}
    for row in manifest['records']:
        if row['status'] == 'annotated':
            previous = refs.setdefault(row['view_id'], row['outcome_sha256'])
            if previous != row['outcome_sha256']:
                raise ValueError('conflicting reference hashes for one view')
    if not refs:
        raise ValueError('no annotated development references')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    total = {}
    for view, checksum in sorted(refs.items()):
        if not re.fullmatch(r'[0-9a-f]{64}', view):
            raise ValueError('invalid view ID')
        name = f'{view}.json'
        packet = json.loads((reference_run / 'packets' / name).read_bytes())
        if packet.pop('view_id') != view or digest(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()) != view:
            raise ValueError('packet digest mismatch')
        candidate = annotate({**packet, 'view_id': view})
        raw_ref = (reference_run / 'outcomes' / name).read_bytes()
        if digest(raw_ref) != checksum:
            raise ValueError('reference digest mismatch')
        ref = json.loads(raw_ref)
        if ref['view_id'] != view or ref['event_id'] != packet['event_id']:
            raise ValueError('reference identity mismatch')
        validate_transcript_annotation(ref, {t['turn_id']: t['text'] for t in packet['turns'] if t['role'] == 'target'})
        score = compare(candidate, ref)
        write_json(output / name, candidate)
        results.append({'view_id': view, 'abstained': not candidate['ideas'],
                        'exact_agreement': all(s['candidate'] == s['matched'] == s['reference'] for s in score.values()),
                        'layers': score})
        for layer, counts in score.items():
            total.setdefault(layer, Counter()).update(counts)
    summary = {'engine': VERSION, 'reference_manifest_sha256': digest(manifest_bytes),
               'reference_views': len(results), 'exact_agreement': sum(r['exact_agreement'] for r in results),
               'abstained': sum(r['abstained'] for r in results), 'layers': total}
    write_json(output / 'evaluation.json', {**summary, 'records': results})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.reference_run, args.output), indent=2))


if __name__ == '__main__':
    main()
