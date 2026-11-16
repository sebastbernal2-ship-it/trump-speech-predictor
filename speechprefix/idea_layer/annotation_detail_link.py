"""Offline detail proposals and diagnostic parent linking with oracle idea labels."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references, score
from .annotation_candidates import propose
from .classical_annotation import words


def rank_parents(detail, context, ideas):
    """Return label IDs in lexical relevance order, abstaining on zero overlap."""
    detail, context = set(words(detail)), set(words(context))
    ranked = []
    for idea in ideas:
        label = set(words(idea['occurrence']))
        if not label:
            continue
        relevance = (len(label & context) + len(label & detail)) / len(label)
        if relevance:
            ranked.append((relevance, idea['idea_id']))
    return sorted(ranked, key=lambda pair: (-pair[0], pair[1]))


def evaluate(root, parser):
    raw = (root / 'manifest.json').read_bytes()
    records = []
    total = Counter()
    for view, packet, turns, ref in load_references(root, raw):
        proposals = defaultdict(set)
        for c in propose(packet, parser):
            if c.kind in {'argument', 'entity', 'clause'}:
                proposals[c.turn_id, c.start, c.end].add((c.parent_start, c.parent_end))
        predictions = {}
        for (turn, a, b), contexts in proposals.items():
            text = turns[turn]
            # All current detail proposals carry the enclosing sentence offsets.
            context = text[slice(*min(contexts))]
            ranked = rank_parents(text[a:b], context, ref['ideas'])
            predictions[turn, a, b] = ranked[0][1] if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]) else None
        mentions = {m['mention_id']: m['evidence'] for m in ref['mentions']}
        point_evidence = defaultdict(set)
        for edge in ref['key_point_mentions']:
            point_evidence[edge['key_point_id']].update(
                (c['turn_id'], c['exact_quote']) for c in mentions[edge['mention_id']])
        parent = {e['key_point_id']: e['idea_id'] for e in ref['key_point_ideas']}
        counts = Counter(points=len(ref['key_points']), proposals=len(proposals),
                         assigned=sum(p is not None for p in predictions.values()))
        for point in ref['key_points']:
            ident = point['key_point_id']
            evidence = point_evidence[ident]
            relevant_turns = {turn for turn, _ in evidence}
            ranked = rank_parents(point['detail'], ' '.join(quote for _, quote in sorted(evidence)), ref['ideas'])
            counts['oracle_detail_top_three'] += parent[ident] in [idea for _, idea in ranked[:3]]
            if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
                counts['oracle_detail_assigned'] += 1
                counts['oracle_detail_correct_parent'] += ranked[0][1] == parent[ident]
            # Require actual detail text agreement, not just overlap with a broad evidence quote.
            detail = tuple(re.findall(r'\w+', point['detail'].casefold()))
            matches = [span for span in proposals if span[0] in relevant_turns and detail and
                       tuple(re.findall(r'\w+', turns[span[0]][span[1]:span[2]].casefold())) == detail and
                       (score({span}, turns, evidence)['overlap_50'] or any(
                           turn == span[0] and any(m.start() <= span[1] and span[2] <= m.start() + len(quote)
                                                    for m in re.finditer(f'(?={re.escape(quote)})', turns[turn]))
                           for turn, quote in evidence))]
            counts['exact_detail_with_evidence'] += bool(matches)
            counts['exact_detail_with_correct_parent'] += any(predictions[s] == parent[ident] for s in matches)
        total.update(counts)
        records.append({'view_id': view, 'counts': counts})
    return {'manifest_sha256': digest(raw), 'reference_views': len(records),
            'oracle_idea_labels': True, 'oracle_detail_metrics_use_reference_text_and_evidence': True, 'totals': total, 'records': records}


def main():
    import spacy

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    parser = spacy.load('en_core_web_sm')
    report = evaluate(args.reference_run, parser)
    report.update(parser=parser.meta['version'], spacy=spacy.__version__)
    write_json(args.output, report)
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
