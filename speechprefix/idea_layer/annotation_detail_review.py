"""Build an offline review queue for candidate detail equivalence, not an accuracy score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .annotation_candidates import propose
from .classical_annotation import words


def rank(detail, candidates, limit):
    """Compare detail text to candidates, without treating overlap as equivalence."""
    target = Counter(words(detail))
    ranked = []
    for span, text in candidates.items():
        tokens = Counter(words(text))
        shared = sum((target & tokens).values())
        f1 = 2 * shared / max(1, sum(target.values()) + sum(tokens.values()))
        ranked.append((f1, span, text))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]


def review_items(detail, candidates, limit):
    return [dict(turn_id=s[0], start=s[1], end=s[2], quote=text, lexical_f1=f1)
            for f1, s, text in rank(detail, candidates, limit)]


def adjacent_windows(bank, turns, max_words=60):
    """Propose two-sentence passages without crossing source turns."""
    by_turn = {}
    for item in bank:
        if item.kind == 'parsed_sentence':
            by_turn.setdefault(item.turn_id, []).append((item.start, item.end))
    windows = {}
    for turn, offsets in by_turn.items():
        ordered = sorted(set(offsets))
        for (start, end), (next_start, next_end) in zip(ordered, ordered[1:]):
            text = turns[turn]
            if end <= next_start and not text[end:next_start].strip():
                quote = text[start:next_end]
                if len(quote.split()) <= max_words:
                    windows[turn, start, next_end] = quote
    return windows


def build(root, parser, limit):
    raw = (root / 'manifest.json').read_bytes()
    rows = []
    views = 0
    for view, packet, turns, ref in load_references(root, raw):
        views += 1
        bank = propose(packet, parser)
        pairs = adjacent_windows(bank, turns)
        mentions = {m['mention_id']: m['evidence'] for m in ref['mentions']}
        parents = {e['key_point_id']: e['idea_id'] for e in ref['key_point_ideas']}
        ideas = {i['idea_id']: i['occurrence'] for i in ref['ideas']}
        for point in ref['key_points']:
            ident = point['key_point_id']
            evidence = [c for e in ref['key_point_mentions'] if e['key_point_id'] == ident
                        for c in mentions[e['mention_id']]]
            windows = [(c['turn_id'], m.start(), m.start() + len(c['exact_quote']))
                       for c in evidence for m in re.finditer(f"(?={re.escape(c['exact_quote'])})", turns[c['turn_id']])]
            candidates = {}
            expanded = {}
            for c in bank:
                if not any(turn == c.turn_id and max(a, c.start) < min(b, c.end) for turn, a, b in windows):
                    continue
                span = c.turn_id, c.start, c.end
                text = turns[c.turn_id][c.start:c.end]
                expanded[span] = text
                if c.kind in {'argument', 'entity', 'clause'}:
                    candidates[span] = text
            with_pairs = {**expanded, **{span: quote for span, quote in pairs.items() if any(
                turn == span[0] and max(a, span[1]) < min(b, span[2]) for turn, a, b in windows)}}
            rows.append(dict(view_id=view, key_point_id=ident, reference_detail=point['detail'],
                             reference_parent=ideas[parents[ident]], evidence=evidence,
                             eligible_spans=len(candidates), review_verdict=None,
                             candidates=review_items(point['detail'], candidates, limit),
                             expanded_candidates=review_items(point['detail'], expanded, limit),
                             window_candidates=review_items(point['detail'], with_pairs, limit)))
    return dict(reference_manifest_sha256=digest(raw), reference_views=views, top_k=limit,
                oracle_evidence_windows=True, records=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--top-k', type=int, default=5)
    args = cli.parse_args()
    if args.top_k < 1:
        cli.error('--top-k must be positive')
    import spacy
    parser = spacy.load('en_core_web_sm')
    report = build(args.reference_run, parser, args.top_k)
    report['parser'] = parser.meta['version']
    write_json(args.output, report)
    print(f"{len(report['records'])} points; {sum(bool(r['candidates']) for r in report['records'])} with eligible candidates")


if __name__ == '__main__':
    main()
