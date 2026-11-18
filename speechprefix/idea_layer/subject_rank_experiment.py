"""Offline ablation of extractive subject pools and sparse/contextual rankers."""
import argparse
from collections import Counter
import json
from itertools import groupby
from pathlib import Path
import time

from .annotation_batch import digest, write_json
from .annotation_candidate_eval import load_references
from .classical_annotation import words
from .parsed_annotation import label_span
from .spoken_evidence import has_spoken_evidence


def proposals(sentence, parser, *, expand=False):
    text = sentence.text.strip()
    noun = label_span(text, parser)
    frame = label_span(text, parser, frames=True, grounded=True)
    candidates = [(kind, label) for kind, label in [('noun', noun), ('frame', frame)] if label]
    for token in sentence:
        if token.dep_ not in {'advcl', 'ccomp', 'xcomp'}:
            continue
        markers = {t.lower_ for t in token.children if t.dep_ in {'mark', 'aux'}}
        if markers & {'so', 'that', 'to', 'because'} and len(sentence) <= 45:
            candidates.append(('relation', text))
    if len(sentence) <= 45 and len(words(text)) >= 2:
        candidates.append(('passage', text))
    if expand:
        for chunk in sentence.doc.noun_chunks:
            if not (sentence.start <= chunk.start < chunk.end <= sentence.end):
                continue
            if chunk.root.pos_ in {'NOUN', 'PROPN'} and words(chunk.text):
                candidates.append(('chunk', chunk.text))
        for token in sentence:
            if token.dep_ not in {'advcl', 'ccomp', 'xcomp', 'conj', 'relcl'}:
                continue
            members = sorted(t.i for t in token.subtree)
            if (members and sentence.start <= members[0] <= members[-1] < sentence.end
                    and len(members) == members[-1] - members[0] + 1 and len(members) <= 32):
                clause = sentence.doc[members[0]:members[-1] + 1].text
                if len(words(clause)) >= 2:
                    candidates.append(('clause', clause))
    unique = {}
    for kind, label in candidates:
        unique.setdefault(label.casefold(), (kind, label))
    return list(unique.values())


def lexical(a, b):
    a, b = set(words(a)), set(words(b))
    return 2 * len(a & b) / max(1, len(a) + len(b))


def features(kind, label, quote):
    return {'kind': kind, 'words': len(words(label)), 'coverage': lexical(label, quote),
            'contains_number': any(c.isdigit() for c in label)}


def embeddings(checkpoint, pairs):
    import torch
    from transformers import AutoModel, AutoTokenizer

    checkpoint = checkpoint.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                      use_safetensors=True).eval()
    if model.config.is_decoder or model.config.is_encoder_decoder or model.config.model_type != 'roberta':
        raise ValueError('expected a local RoBERTa encoder checkpoint')
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(pairs), 16):
            context, labels = zip(*pairs[start:start + 16])
            inputs = tokenizer(list(context), list(labels), padding=True, truncation='only_first',
                               max_length=256, return_tensors='pt')
            vectors.extend(model(**inputs).last_hidden_state[:, 0].numpy())
    return vectors


def evaluate(root, parser, checkpoint=None, *, expand=False, windows=False):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    import numpy as np
    from scipy.sparse import hstack

    raw = (root / 'manifest.json').read_bytes()
    rows, evidence_keys = [], set()
    for view, packet, _, ref in load_references(root, raw):
        labels = {i['idea_id']: i['occurrence'] for i in ref['ideas']}
        mentions = {m['mention_id']: m['evidence'] for m in ref['mentions']}
        grounded = [(e['idea_id'], c['turn_id'], c['exact_quote']) for e in ref['idea_mentions']
                    for c in mentions[e['mention_id']]]
        evidence_keys.update((view, i) for i in labels)
        prior = ''
        for turn in packet['turns']:
            if turn['role'] == 'target' and has_spoken_evidence(turn['text']):
                previous = None
                for sent in parser(turn['text']).sents:
                    quote = sent.text.strip()
                    if not has_spoken_evidence(quote):
                        continue
                    targets = {i: labels[i] for i, t, q in grounded if t == turn['turn_id'] and
                               (q in quote or quote in q)}
                    key = view, turn['turn_id'], sent.start_char
                    candidates = [(kind, label, targets) for kind, label in proposals(sent, parser, expand=expand)]
                    if windows and previous is not None:
                        combined = turn['text'][previous.start_char:sent.end_char].strip()
                        if len(combined.split()) <= 60:
                            window_targets = {i: labels[i] for i, t, q in grounded if t == turn['turn_id']
                                              and (q in combined or combined in q)}
                            candidates.append(('window', combined, window_targets))
                    for kind, label, candidate_targets in candidates:
                        evidence_start = previous.start_char if kind == 'window' else sent.start_char
                        rows.append(dict(key=key, evidence_start=evidence_start, evidence_end=sent.end_char,
                                         event=packet['event_id'], kind=kind, label=label,
                                         quote=quote, context=f'{quote}\nPrevious turn: {prior[-500:]}',
                                         proxy={i: lexical(label, target) for i, target in candidate_targets.items()}))
                    previous = sent
            prior = turn['text']
    started = time.perf_counter()
    dense = embeddings(checkpoint, [(r['context'], r['label']) for r in rows]) if checkpoint else None
    encoder_seconds = time.perf_counter() - started if checkpoint else None
    features_by_row = [features(r['kind'], r['label'], r['quote']) for r in rows]
    y = np.array([max(r['proxy'].values(), default=0.) for r in rows])
    totals = Counter()
    rank_seconds = Counter()
    covered = {}
    selected = []

    def record(method, indices):
        for index in indices:
            r = rows[index]
            totals[method] += 1
            covered.setdefault(method, set()).update((r['key'][0], i) for i, value in r['proxy'].items() if value >= .5)
            if method not in {'pool', 'noun_pool'}:
                selected.append(dict(method=method, view_id=r['key'][0], turn_id=r['key'][1],
                                     start=r['evidence_start'], end=r['evidence_end'], label=r['label'], kind=r['kind']))

    record('pool', range(len(rows)))
    record('noun_pool', [i for i, r in enumerate(rows) if r['kind'] == 'noun'])
    for train, test in GroupKFold(5).split(rows, groups=[r['event'] for r in rows]):
        vectorizer = DictVectorizer()
        x_train = vectorizer.fit_transform([features_by_row[i] for i in train])
        x_test = vectorizer.transform([features_by_row[i] for i in test])
        matrices = {'sparse': (x_train, x_test)}
        if dense is not None:
            vectors = np.asarray(dense)
            matrices['contextual'] = (hstack([x_train, vectors[train]]).tocsr(),
                                      hstack([x_test, vectors[test]]).tocsr())
        predictions = {}
        for name, (a, b) in matrices.items():
            model = Ridge(alpha=10).fit(a, y[train])
            start = time.perf_counter()
            predictions[name] = model.predict(b)
            rank_seconds[name] += time.perf_counter() - start
        predictions['syntax_first'] = [dict(relation=3, frame=2, noun=1, passage=0, clause=-1, chunk=-2, window=-3)[rows[i]['kind']] for i in test]
        for method, scores in predictions.items():
            ordered = sorted(zip(test, scores), key=lambda x: (rows[x[0]]['key'], -x[1], x[0]))
            groups = [[i for i, _ in group] for _, group in groupby(ordered, key=lambda x: rows[x[0]]['key'])]
            record(method, [group[0] for group in groups])
            if expand:
                budgeted = []
                for group in groups:
                    first = group[0]
                    budgeted.append(first)
                    second = next((i for i in group[1:] if lexical(rows[i]['label'], rows[first]['label']) < .8), None)
                    if second is not None:
                        budgeted.append(second)
                record(f'{method}_diverse_top2', budgeted)
    pool_hits = covered.get('pool', set())
    sparse_hits = covered.get('sparse', set())
    return dict(reference_manifest_sha256=digest(raw), folds=5, reference_ideas=len(evidence_keys),
                sparse_gap=dict(no_pool_proxy_hit=len(evidence_keys - pool_hits),
                                pool_hit_not_selected=len(pool_hits - sparse_hits),
                                selected_proxy_hit=len(sparse_hits)),
                expanded=expand, windows=windows, encoder_seconds=encoder_seconds, ranking_predict_seconds=dict(rank_seconds),
                idea_diagnostics=[dict(view_id=view, reference_id=idea,
                                       proxy_hits=[name for name in totals if (view, idea) in covered.get(name, ())])
                                  for view, idea in sorted(evidence_keys)],
                checkpoint=checkpoint.name if checkpoint else None,
                methods={method: dict(emitted=count, reference_ideas_with_lexical_proxy_hit=len(covered.get(method, ())))
                         for method, count in totals.items()}, selections=selected)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--expand', action='store_true')
    cli.add_argument('--windows', action='store_true')
    args = cli.parse_args()
    import spacy
    report = evaluate(args.reference_run, spacy.load('en_core_web_sm', disable=['ner']), args.checkpoint, expand=args.expand, windows=args.windows)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'selections', 'idea_diagnostics'}}, indent=2))


if __name__ == '__main__':
    main()
