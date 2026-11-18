"""Measure offline span proposal recall against frozen transcript references."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidates import propose
from .transcript_annotations import validate_transcript_annotation

VARIANTS = {
    'sentences': {'baseline_sentence'},
    'parsed_clauses': {'baseline_sentence', 'parsed_sentence', 'clause'},
    'all': {'baseline_sentence', 'parsed_sentence', 'clause', 'argument', 'entity'},
}


def score(candidates, turns, references):
    """Reference-centric oracle recall, not precision or a graph score."""
    counts = Counter(candidates=len(candidates), reference=len(references), exact=0, overlap_50=0)
    bank = defaultdict(set)
    for turn, start, end in candidates:
        bank[turn].add((start, end))
    for turn, quote in references:
        exact = overlap = False
        for match in re.finditer(f'(?={re.escape(quote)})', turns[turn]):
            a, b = match.start(), match.start() + len(quote)
            for start, end in bank[turn]:
                exact |= (start, end) == (a, b)
                intersection = min(end, b) - max(start, a)
                overlap |= 2 * intersection >= max(end, b) - min(start, a)
        counts.update(exact=int(exact), overlap_50=int(overlap))
    return counts


def evaluate(root, parser):
    raw = (root / 'manifest.json').read_bytes()
    refs = {}
    for row in json.loads(raw)['records']:
        if row['status'] == 'annotated':
            checksum = refs.setdefault(row['view_id'], row['outcome_sha256'])
            if checksum != row['outcome_sha256']:
                raise ValueError('conflicting reference hashes')
    if not refs:
        raise ValueError('no frozen references')
    totals = defaultdict(Counter)
    records = []
    for view, checksum in sorted(refs.items()):
        if not re.fullmatch(r'[a-f0-9]{64}', view):
            raise ValueError('invalid view id')
        name = f'{view}.json'
        packet = json.loads((root / 'packets' / name).read_bytes())
        if packet.pop('view_id') != view or digest(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()) != view:
            raise ValueError('packet digest mismatch')
        ref_raw = (root / 'outcomes' / name).read_bytes()
        if digest(ref_raw) != checksum:
            raise ValueError('reference digest mismatch')
        turns = {t['turn_id']: t['text'] for t in packet['turns'] if t['role'] == 'target'}
        ref = validate_transcript_annotation(json.loads(ref_raw), turns)
        if ref['view_id'] != view or ref['event_id'] != packet['event_id']:
            raise ValueError('reference identity mismatch')
        mentions = {m['mention_id']: m['evidence'] for m in ref['mentions']}
        point_mentions = {edge['mention_id'] for edge in ref['key_point_mentions']}
        evidence = lambda ids: {(c['turn_id'], c['exact_quote']) for i in ids for c in mentions[i]}
        candidates = propose(packet, parser)
        row = {'view_id': view, 'scores': {}}
        for variant, kinds in VARIANTS.items():
            spans = {(c.turn_id, c.start, c.end) for c in candidates if c.kind in kinds}
            for subset, ids in [('all_evidence', mentions), ('point_evidence', point_mentions)]:
                key = f'{variant}/{subset}'
                row['scores'][key] = score(spans, turns, evidence(ids))
                totals[key].update(row['scores'][key])
        records.append(row)
    return {'reference_views': len(refs), 'manifest_sha256': digest(raw),
            'parser': {key: parser.meta.get(key) for key in ('lang', 'name', 'version', 'spacy_version')},
            'totals': totals, 'records': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import spacy
    report = evaluate(args.reference_run, spacy.load('en_core_web_sm'))
    report['spacy_runtime_version'] = spacy.__version__
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
