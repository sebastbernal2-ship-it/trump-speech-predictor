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
                                  source_run=idea.get('source_run'),
                                  lexical_overlap=overlap, verdict=None))
        pairs.sort(key=lambda p: (-p['lexical_overlap'], not p['shared_turn'], p['candidate_id']))
        yield dict(reference_id=rid, reference_label=ref['occurrence'],
                   reference_evidence=reference_evidence[rid], eligible_pairs=len(pairs), pairs=pairs[:limit])


def merge_candidates(graphs):
    """Pool distinct subject hypotheses while preserving their source evidence."""
    pooled = dict(ideas=[], mentions=[], idea_mentions=[])
    seen = set()
    for name, graph in graphs:
        evidence = evidence_by_idea(graph)
        for idea in graph['ideas']:
            label = idea['occurrence']
            cites = {(c['turn_id'], c['exact_quote']) for c in evidence[idea['idea_id']]}
            key = label.casefold(), frozenset(cites)
            if key in seen:
                continue
            seen.add(key)
            ident = f"proposal_{len(pooled['ideas']) + 1}"
            pooled['ideas'].append(dict(idea_id=ident, occurrence=label, source_run=name))
            pooled['mentions'].append(dict(mention_id=ident, evidence=[
                dict(turn_id=turn, exact_quote=quote) for turn, quote in sorted(cites)]))
            pooled['idea_mentions'].append(dict(idea_id=ident, mention_id=ident))
    return pooled


def build(reference_run, candidate_run, limit, alternatives=()):
    raw = (reference_run / 'manifest.json').read_bytes()
    rows, hashes = [], {}
    for view, packet, turns, reference in load_references(reference_run, raw):
        candidate_raw = (candidate_run / f'{view}.json').read_bytes()
        candidate = validate_transcript_annotation(json.loads(candidate_raw), turns)
        if candidate.get('view_id') != view or candidate['event_id'] != packet['event_id']:
            raise ValueError('candidate identity mismatch')
        hashes[view] = digest(candidate_raw)
        graphs = [(candidate_run.name, candidate)]
        for index, path in enumerate(alternatives, 1):
            alternative_raw = (path / f'{view}.json').read_bytes()
            alternative = validate_transcript_annotation(json.loads(alternative_raw), turns)
            if alternative.get('view_id') != view or alternative['event_id'] != packet['event_id']:
                raise ValueError('alternative candidate identity mismatch')
            hashes[f'{index}:{view}'] = digest(alternative_raw)
            graphs.append((path.name, alternative))
        pool = merge_candidates(graphs) if alternatives else candidate
        rows.extend(dict(view_id=view, **row) for row in shortlist(pool, reference, limit))
    return dict(reference_manifest_sha256=digest(raw), candidate_sha256=hashes, top_k=limit,
                verdicts=['same_subject', 'broader', 'narrower', 'different', 'uncertain'], records=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--candidate-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--alternative-run', action='append', type=Path, default=[])
    cli.add_argument('--top-k', type=int, default=3)
    args = cli.parse_args()
    if args.top_k < 1:
        cli.error('--top-k must be positive')
    report = build(args.reference_run, args.candidate_run, args.top_k, args.alternative_run)
    write_json(args.output, report)
    rows = report['records']
    print(f'{len(rows)} unmatched references; {sum(bool(r["pairs"]) for r in rows)} with review pairs; '
          f'{sum(len(r["pairs"]) for r in rows)} pairs queued')


if __name__ == '__main__':
    main()
