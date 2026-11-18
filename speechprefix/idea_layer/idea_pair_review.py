"""Export local idea-alignment candidates for review, not automatic agreement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .idea_agreement import align_ideas
from .transcript_annotations import validate_transcript_annotation


def evidence_by_idea(graph):
    mentions = {m['mention_id']: m['evidence'] for m in graph['mentions']}
    result = defaultdict(list)
    for edge in graph['idea_mentions']:
        result[edge['idea_id']].extend(mentions[edge['mention_id']])
    return result


def shortlist(candidate, reference, limit):
    aligned = align_ideas(candidate['ideas'], reference['ideas'])
    candidate_evidence, reference_evidence = map(evidence_by_idea, (candidate, reference))
    for ref in reference['ideas']:
        rid = ref['idea_id']
        if rid not in aligned['unmatched_reference_ids']:
            continue
        ref_words = set(re.findall(r'\w+', ref['occurrence'].casefold()))
        ref_turns = {c['turn_id'] for c in reference_evidence[rid]}
        pairs = []
        for idea in candidate['ideas']:
            cid = idea['idea_id']
            words = set(re.findall(r'\w+', idea['occurrence'].casefold()))
            overlap = len(words & ref_words) / max(1, len(words | ref_words))
            shared_turn = bool(ref_turns & {c['turn_id'] for c in candidate_evidence[cid]})
            if overlap or shared_turn:
                pairs.append(dict(candidate_id=cid, candidate_label=idea['occurrence'],
                                  candidate_evidence=candidate_evidence[cid], shared_turn=shared_turn,
                                  lexical_overlap=overlap, verdict=None))
        pairs.sort(key=lambda p: (-p['lexical_overlap'], not p['shared_turn'], p['candidate_id']))
        yield dict(reference_id=rid, reference_label=ref['occurrence'],
                   reference_evidence=reference_evidence[rid], eligible_pairs=len(pairs), pairs=pairs[:limit])


def build(reference_run, candidate_run, limit):
    raw = (reference_run / 'manifest.json').read_bytes()
    rows, hashes = [], {}
    for view, packet, turns, reference in load_references(reference_run, raw):
        candidate_raw = (candidate_run / f'{view}.json').read_bytes()
        candidate = validate_transcript_annotation(json.loads(candidate_raw), turns)
        if candidate.get('view_id') != view or candidate['event_id'] != packet['event_id']:
            raise ValueError('candidate identity mismatch')
        hashes[view] = digest(candidate_raw)
        rows.extend(dict(view_id=view, **row) for row in shortlist(candidate, reference, limit))
    return dict(reference_manifest_sha256=digest(raw), candidate_sha256=hashes, top_k=limit,
                verdicts=['same_subject', 'broader', 'narrower', 'different', 'uncertain'], records=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--candidate-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--top-k', type=int, default=3)
    args = cli.parse_args()
    if args.top_k < 1:
        cli.error('--top-k must be positive')
    report = build(args.reference_run, args.candidate_run, args.top_k)
    write_json(args.output, report)
    rows = report['records']
    print(f'{len(rows)} unmatched references; {sum(bool(r["pairs"]) for r in rows)} with review pairs; '
          f'{sum(len(r["pairs"]) for r in rows)} pairs queued')


if __name__ == '__main__':
    main()
