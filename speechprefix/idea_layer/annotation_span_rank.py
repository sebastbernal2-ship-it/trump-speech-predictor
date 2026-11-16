"""Offline, event-grouped ranking of evidence and detail-evidence spans."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references, score
from .annotation_candidates import propose


def examples(packet, turns, parser):
    groups = defaultdict(set)
    for c in propose(packet, parser):
        groups[c.turn_id, c.start, c.end].add(c.kind)
    rows = []
    for span, kinds in groups.items():
        turn, start, end = span
        text = turns[turn][start:end]
        features = {f'kind={kind}': 1 for kind in kinds}
        features.update({f'word={w}': 1 for w in re.findall(r'\w+', text.casefold())})
        features.update(length_bucket=str(min(8, len(text.split()) // 4)),
                        has_number=int(any(c.isdigit() for c in text)),
                        terminal_punctuation=int(text.endswith(('.', '!', '?'))))
        rows.append((span, features))
    return rows


def evaluate(root, parser, budget):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    raw = (root / 'manifest.json').read_bytes()
    views = []
    for view, packet, turns, ref in load_references(root, raw):
        detail = {edge['mention_id'] for edge in ref['key_point_mentions']}
        targets = {task: {(c['turn_id'], c['exact_quote']) for m in ref['mentions']
                          if task == 'evidence' or m['mention_id'] in detail for c in m['evidence']}
                   for task in ('evidence', 'detail_evidence')}
        views.append((view, packet['event_id'], turns, examples(packet, turns, parser), targets))
    groups = [v[1] for v in views]
    folds = min(5, len(set(groups)))
    if folds < 2:
        raise ValueError('at least two distinct reference events required')
    totals = defaultdict(Counter)
    records = []
    for fold, (train, test) in enumerate(GroupKFold(folds).split(views, groups=groups)):
        for task in ('evidence', 'detail_evidence'):
            x, y = [], []
            for i in train:
                _, _, turns, rows, targets = views[i]
                for span, features in rows:
                    x.append(features)
                    y.append(bool(score({span}, turns, targets[task])['overlap_50']))
            if len(set(y)) != 2:
                raise ValueError(f'fold {fold}/{task}: training requires positive and negative spans')
            model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000, class_weight='balanced'))
            model.fit(x, y)
            for i in test:
                view, event, turns, rows, targets = views[i]
                probabilities = model.predict_proba([f for _, f in rows])[:, 1] if rows else []
                ranked = sorted(zip(probabilities, (s for s, _ in rows)), key=lambda x: (-x[0], x[1]))
                # Equal per-view budgets; deterministic parser-kind priority comparator.
                priority = sorted(rows, key=lambda r: (not bool({'kind=baseline_sentence', 'kind=parsed_sentence'} & r[1].keys()), r[0]))
                for method, spans in [('learned', {s for _, s in ranked[:budget]}),
                                      ('sentence_first', {s for s, _ in priority[:budget]})]:
                    counts = score(spans, turns, targets[task])
                    totals[f'{task}/{method}'].update(counts)
                    records.append(dict(view_id=view, event_id=event, fold=fold, task=task,
                                        method=method, scores=counts, spans=sorted(spans)))
    return dict(manifest_sha256=digest(raw), folds=folds, reference_views=len(views),
                budget=budget, totals=dict(totals), records=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget', type=int, default=8)
    args = p.parse_args()
    if args.budget < 1:
        p.error('budget must be positive')
    import spacy
    import sklearn
    parser = spacy.load('en_core_web_sm')
    result = evaluate(args.reference_run, parser, args.budget)
    result.update(parser=parser.meta['version'], spacy=spacy.__version__, sklearn=sklearn.__version__)
    write_json(args.output, result)
    print(json.dumps(result['totals'], indent=2))


if __name__ == '__main__':
    main()
