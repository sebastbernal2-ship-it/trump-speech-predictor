"""Score held-out generations against the frozen references with a local embedding matcher.

An item counts as matched when its cosine similarity to some item in the other set reaches the
threshold, so differently worded but same-meaning ideas match. Recall = reference items matched;
precision = generated items matched. Mention quotes are checked for exact presence in the transcript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT = {'ideas': 'occurrence', 'key_points': 'detail', 'topics': 'label', 'opinions': 'text', 'emotions': 'label'}


def embed(model_dir: str, texts: list[str]):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    out = []
    for i in range(0, len(texts), 64):
        batch = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            h = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        v = (h * mask).sum(1) / mask.sum(1)
        out.append(torch.nn.functional.normalize(v, dim=-1))
    return torch.cat(out) if out else torch.zeros(0, 384)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='fold output directories')
    p.add_argument('--embedder', required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.6, 0.7])
    a = p.parse_args()
    rows = {r['view_id']: r for r in map(json.loads, a.pairs.read_text().splitlines())}
    totals = {layer: {t: [0, 0, 0, 0] for t in a.thresholds} for layer in TEXT}  # ref_hit, ref_n, gen_hit, gen_n
    quotes = [0, 0]; parsed = [0, 0]
    for run in a.runs:
        for path in sorted(run.glob('*.json')):
            d = json.loads(path.read_text()); row = rows[d['view_id']]; ref = json.loads(row['target'])
            parsed[1] += 1
            try:
                gen = json.loads(d['raw']); parsed[0] += 1
            except json.JSONDecodeError:
                gen = {}
            for m in gen.get('mentions', []):
                for e in m.get('evidence', []):
                    quotes[1] += 1; quotes[0] += e.get('exact_quote', '') in row['input']
            for layer, key in TEXT.items():
                r = [x[key] for x in ref.get(layer, []) if key in x]
                g = [x[key] for x in gen.get(layer, []) if isinstance(x, dict) and key in x]
                if not r:
                    continue
                if not g:
                    for t in a.thresholds:
                        totals[layer][t][1] += len(r)
                    continue
                e = embed(a.embedder, r + g); sim = e[:len(r)] @ e[len(r):].T
                for t in a.thresholds:
                    c = totals[layer][t]
                    c[0] += int((sim.max(1).values >= t).sum()); c[1] += len(r)
                    c[2] += int((sim.max(0).values >= t).sum()); c[3] += len(g)
    out = {'parsed_json': parsed, 'quotes_exact': quotes, 'layers': {}}
    for layer, by in totals.items():
        out['layers'][layer] = {str(t): {'recall': round(c[0] / c[1], 3) if c[1] else None,
                                        'precision': round(c[2] / c[3], 3) if c[3] else None,
                                        'ref': c[1], 'gen': c[3]} for t, c in by.items()}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
