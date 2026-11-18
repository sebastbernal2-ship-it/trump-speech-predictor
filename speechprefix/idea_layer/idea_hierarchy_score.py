"""Summarize reviewed idea-scale pairs without conflating them with coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .annotation_batch import digest, write_json

FIELDS = ('subject_validity', 'scale_relation_to_parent', 'attachment_verdict')


def pair_key(row):
    return (row['view_id'], row['inventory_index'],
            row.get('reference_key_point_id') or row['reference_key_point']['key_point_id'])


def summarize(queue, audit):
    pairs = {pair_key(row): row for row in queue['records']}
    if len(pairs) != len(queue['records']):
        raise ValueError('duplicate pair in queue')
    seen = set()
    counts = {field: Counter() for field in FIELDS}
    for row in audit['records']:
        key = pair_key(row)
        if key not in pairs or key in seen:
            raise ValueError('unknown or duplicate reviewed pair')
        seen.add(key)
        for field in FIELDS:
            if row[field] not in queue['verdicts'][field]:
                raise ValueError(f'invalid {field}')
        counts['subject_validity'][row['subject_validity']] += 1
        if row['subject_validity'] == 'valid':
            for field in FIELDS[1:]:
                counts[field][row[field]] += 1
    return {'unit': 'reviewed_pair', 'queued': len(pairs), 'reviewed': len(seen),
            'unreviewed': len(pairs) - len(seen), 'counts': counts}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('queue', 'audit', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    args = cli.parse_args()
    queue_raw, audit_raw = args.queue.read_bytes(), args.audit.read_bytes()
    audit = json.loads(audit_raw)
    if audit['queue_sha256'] != digest(queue_raw):
        raise ValueError('audit queue hash mismatch')
    report = summarize(json.loads(queue_raw), audit)
    write_json(args.output, {**report, 'queue_sha256': digest(queue_raw), 'audit_sha256': digest(audit_raw)})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
