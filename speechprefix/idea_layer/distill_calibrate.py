"""Calibrate per-layer cosine thresholds against judged calibration pairs.

Merges judge files (pair_id, label in same/broader/narrower/different) with calibration pairs
(pair_id, layer, cosine). For each layer and overall, sweeps thresholds 0.30..0.90 (step 0.05) and
reports precision/recall/F1 of 'cosine >= t' predicting positive. Two positive definitions:
strict (label == same) and multi-scale (same, broader or narrower). Best-F1 threshold per layer is
chosen for each definition (ties go to the lower threshold).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ('same', 'broader', 'narrower', 'different')
POSITIVE = {'strict': {'same'}, 'multi': {'same', 'broader', 'narrower'}}
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]


def prf(rows, positive: set[str], t: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = r['cosine'] >= t
        gold = r['label'] in positive
        if pred and gold: tp += 1
        elif pred: fp += 1
        elif gold: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'threshold': t, 'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def best(rows, positive: set[str]) -> dict:
    sweep = [prf(rows, positive, t) for t in THRESHOLDS]
    top = max(sweep, key=lambda s: (s['f1'], -s['threshold']))
    return {'best': top, 'sweep': sweep}


def load_merged(judge_paths: list[Path], pairs_path: Path) -> list[dict]:
    pairs = {p['pair_id']: p for p in json.loads(pairs_path.read_text())}
    judged: dict[str, str] = {}
    for path in judge_paths:
        for j in json.loads(path.read_text()):
            if j['label'] not in LABELS:
                raise ValueError(f"unknown label {j['label']!r} for {j['pair_id']}")
            if j['pair_id'] in judged:
                raise ValueError(f"duplicate judgement for {j['pair_id']}")
            judged[j['pair_id']] = j['label']
    missing = sorted(set(pairs) - set(judged))
    extra = sorted(set(judged) - set(pairs))
    if missing or extra:
        raise ValueError(f'pair mismatch: {len(missing)} unjudged, {len(extra)} unknown')
    return [{'pair_id': pid, 'layer': pairs[pid]['layer'], 'cosine': pairs[pid]['cosine'], 'label': judged[pid]}
            for pid in sorted(pairs)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--judges', type=Path, nargs='+', required=True)
    p.add_argument('--pairs', type=Path, required=True, help='calib-pairs.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = load_merged(a.judges, a.pairs)
    layers = sorted({r['layer'] for r in rows})
    report = {'n_pairs': len(rows), 'label_counts': {}, 'thresholds': THRESHOLDS, 'layers': {}, 'overall': {}}
    report['label_counts']['overall'] = dict(Counter(r['label'] for r in rows))
    for layer in layers:
        sub = [r for r in rows if r['layer'] == layer]
        entry = {'n': len(sub), 'label_counts': dict(Counter(r['label'] for r in sub))}
        for name, pos in POSITIVE.items():
            entry[name] = best(sub, pos)
        report['layers'][layer] = entry
    report['overall'] = {'n': len(rows), 'label_counts': report['label_counts']['overall']}
    for name, pos in POSITIVE.items():
        report['overall'][name] = best(rows, pos)
    report['chosen_thresholds'] = {
        name: {layer: report['layers'][layer][name]['best']['threshold'] for layer in layers} for name in POSITIVE}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'n_pairs': report['n_pairs'], 'chosen_thresholds': report['chosen_thresholds'],
                      'best': {layer: {name: report['layers'][layer][name]['best'] for name in POSITIVE}
                               for layer in layers}}, indent=1))


if __name__ == '__main__':
    main()
