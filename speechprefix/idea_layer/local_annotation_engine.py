"""Deterministic transcript-only annotation baseline; no inference service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .transcript_annotations import EDGES, LAYERS, validate_transcript_annotation

VERSION = 'local-rules-v1'
SUBJECTS = (
    ('Border and immigration enforcement', 'Immigration', r'border|immigra\w*|deport\w*|asylum'),
    ('Trade tariffs', 'Economy', r'tariffs?|import duties'),
    ('Inflation and prices', 'Economy', r'inflation|prices|cost of living'),
    ('Ukraine war', 'Foreign affairs', r'ukraine|ukrainian'),
    ('Israel and Gaza conflict', 'Foreign affairs', r'gaza|hamas|ceasefire'),
    ('Elections and voting', 'Elections', r'voting|ballots?|elections?|polling'),
    ('Energy production', 'Energy', r'oil production|drilling|natural gas|energy production'),
    ('Federal taxation', 'Economy', r'taxes|tax cuts?|tax rates?'),
)
CUES = (
    ('supports', r'\bI (?:support|endorse|favor)\b'),
    ('opposes', r'\bI (?:oppose|reject|do not support|don.t support)\b'),
    ('proposes', r'\bI propose\b'),
    ('requests', r'\bI (?:ask|urge|call on)\b'),
    ('forecasts', r'\bI (?:predict|expect)\b'),
    ('asserts', r'\bI (?:believe|think)\b'),
)


def annotate(packet: dict) -> dict:
    """Build a linked graph using only the input packet and fixed rules."""
    row = dict(artifact_type='transcript-annotation', schema_version=1, coverage='full_pass',
               event_id=packet['event_id'], view_id=packet['view_id'], annotation_pass='local_engine_candidate',
               generator={'engine': VERSION}, review_status='pending_user_review',
               **{key: [] for key in (*LAYERS, *EDGES, 'opinion_targets')})

    def add(layer, prefix, **fields):
        ident = f'{prefix}{len(row[layer]) + 1}'
        row[layer].append({LAYERS[layer]: ident, **fields})
        return ident

    def link(relation, left, right):
        a, b = EDGES[relation]
        row[relation].append({LAYERS[a]: left, LAYERS[b]: right})

    topics, ideas, turns = {}, {}, {}
    for turn in packet['turns']:
        if turn['role'] != 'target':
            continue
        source, text = turn['turn_id'], turn['text']
        turns[source] = text
        for part in re.finditer(r'[^.!?\n]+[.!?]*', text):
            quote = part[0].strip()
            matched = [(subject, topic) for subject, topic, rule in SUBJECTS
                       if re.search(rf'\b(?:{rule})\b', quote, re.I)]
            if not matched:
                continue
            mention = add('mentions', 'm', evidence=[{'turn_id': source, 'exact_quote': quote}])
            for subject, topic in matched:
                if subject not in ideas:
                    ideas[subject] = add('ideas', 'i', occurrence=subject,
                                         uncertainty='Keyword match; subject boundaries may be broader than the passage.')
                    if topic not in topics:
                        topics[topic] = add('topics', 't', label=topic)
                    link('topic_ideas', topics[topic], ideas[subject])
                link('idea_mentions', ideas[subject], mention)
            # Abstain on attachment when several subjects match the same span.
            if len(matched) == 1:
                idea = ideas[matched[0][0]]
                for stance, cue in CUES:
                    if re.search(cue, quote, re.I):
                        opinion = add('opinions', 'o', attribution='speaker', stance=stance,
                                      text=quote, uncertainty='Cue match; scope and intent require review.')
                        link('opinion_mentions', opinion, mention)
                        row['opinion_targets'].append(dict(opinion_id=opinion, target_type='idea', target_id=idea))
                        break
    if not ideas:
        row['empty_reason'] = 'No supported subject rule fired; this is an abstention, not a finding of no substantive content.'
    return validate_transcript_annotation(row, turns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experimental-bulk', action='store_true')
    args = parser.parse_args()
    if not args.experimental_bulk:
        parser.error('baseline is not rollout-ready; use the development evaluator first')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.intake / 'manifest.json').read_bytes())
    views = sorted({r['view_id'] for r in manifest['records'] if 'view_id' in r})
    results = []
    for view in views:
        name = f'{view}.json'
        packet = json.loads((args.intake / 'packets' / name).read_bytes())
        actual = packet.pop('view_id')
        if actual != view or digest(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()) != view:
            raise ValueError('packet digest mismatch')
        result = annotate({**packet, 'view_id': view})
        write_json(args.output / name, result)
        results.append({'view_id': view, 'status': 'candidate' if result['ideas'] else 'abstained',
                        'sha256': digest((args.output / name).read_bytes())})
    write_json(args.output / 'engine-report.json', {'engine': VERSION, 'records': results})


if __name__ == '__main__':
    main()
