"""Integrate offline detail retrieval with transcript-only idea extraction."""
import argparse
from functools import partial
import json
from pathlib import Path

from .annotation_candidates import propose
from .annotation_detail_selector import select
from .classical_annotation import annotate as classical_annotate, words
from .local_annotation_evaluate import evaluate
from .transcript_annotations import validate_transcript_annotation

VERSION = 'classical-parent-details-v1'


def annotate(packet, *, parser, budget=2):
    """Attach literal sentence details to extracted parents, without draft inputs."""
    row = classical_annotate(packet)
    turns = {t['turn_id']: t['text'] for t in packet['turns'] if t['role'] == 'target'}
    bank = propose(packet, parser)
    mentions = {}
    for mention in row['mentions']:
        for cite in mention['evidence']:
            mentions.setdefault((cite['turn_id'], cite['exact_quote']), mention['mention_id'])
    idea_edges = {(e['idea_id'], e['mention_id']) for e in row['idea_mentions']}
    details = {p['key_point_id']: p['detail'] for p in row['key_points']}
    existing = {(e['idea_id'], details[e['key_point_id']]) for e in row['key_point_ideas']}
    for idea in row['ideas']:
        ident, label = idea['idea_id'], idea['occurrence']
        label_words = set(words(label))
        for turn, start, end in sorted(select(bank, turns, label, budget, conditioned=True)):
            quote = turns[turn][start:end]
            tokens = set(words(quote))
            if not (tokens & label_words and tokens - label_words) or (ident, quote) in existing:
                continue
            mention = mentions.get((turn, quote))
            if mention is None:
                mention = f"mention_{len(row['mentions']) + 1}"
                row['mentions'].append(dict(mention_id=mention, evidence=[dict(turn_id=turn, exact_quote=quote)]))
                mentions[turn, quote] = mention
            if (ident, mention) not in idea_edges:
                row['idea_mentions'].append(dict(idea_id=ident, mention_id=mention))
                idea_edges.add((ident, mention))
            point = f"key_point_{len(row['key_points']) + 1}"
            row['key_points'].append(dict(key_point_id=point, detail=quote,
                                          uncertainty='Extractive candidate; parent and specificity require review.'))
            row['key_point_mentions'].append(dict(key_point_id=point, mention_id=mention))
            row['key_point_ideas'].append(dict(key_point_id=point, idea_id=ident))
            existing.add((ident, quote))
    row['generator'] = {'engine': VERSION, 'parser': parser.meta['version'], 'budget': budget}
    return validate_transcript_annotation(row, turns)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    import spacy
    annotate_local = partial(annotate, parser=spacy.load('en_core_web_sm'))
    print(json.dumps(evaluate(args.reference_run, args.output, annotator=annotate_local, engine=VERSION), indent=2))


if __name__ == '__main__':
    main()
