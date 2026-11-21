"""Export stratified (reference item, best-matching generated item, cosine) pairs for calibrating semantic equivalence.

For every held-out view with parsable generated JSON, each reference item in each layer is paired with its
highest-cosine generated item (MiniLM embeddings, same text fields as distill_eval.TEXT). Per layer the pairs are
sorted by cosine and up to PER_LAYER pairs are taken evenly spaced over that order, so the sample spans the full
similarity range. The combined sample is shuffled with a fixed seed and written with sequential pair ids.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from speechprefix.idea_layer.distill_eval import TEXT, embed

BASE = Path('/home/vorte/code/trump-speech-predictor/data/source-archives/speech-content/outcomes/distill-v1')
EMBEDDER = '/home/vorte/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PER_LAYER = 80
TRANSCRIPT_CHARS = 6000
SEED = 0


def items(obj: dict, layer: str) -> list[str]:
    key = TEXT[layer]
    out = []
    for x in obj.get(layer, []) or []:
        if isinstance(x, dict) and isinstance(x.get(key), str) and x[key].strip():
            out.append(x[key])
    return out


def main() -> None:
    rows = [json.loads(line) for line in (BASE / 'pairs.jsonl').read_text().splitlines() if line.strip()]
    # Collect (view, layer) candidate sets first so every text is embedded in one batched pass.
    cases = []  # (row, layer, ref_items, gen_items)
    skipped = 0
    for row in rows:
        ref = json.loads(row['target'])
        gen_path = next((p for p in (BASE / f'fold{k}' / f"{row['view_id']}.json" for k in range(5)) if p.exists()), None)
        if gen_path is None:
            continue
        raw = json.loads(gen_path.read_text()).get('raw')
        try:
            gen = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            skipped += 1
            continue
        if not isinstance(gen, dict):
            skipped += 1
            continue
        for layer in TEXT:
            r, g = items(ref, layer), items(gen, layer)
            if r and g:
                cases.append((row, layer, r, g))

    texts = sorted({t for _, _, r, g in cases for t in r + g})
    index = {t: i for i, t in enumerate(texts)}
    vecs = embed(EMBEDDER, texts) if texts else None

    by_layer: dict[str, list[dict]] = {layer: [] for layer in TEXT}
    for row, layer, r, g in cases:
        rv = vecs[[index[t] for t in r]]
        gv = vecs[[index[t] for t in g]]
        sim = rv @ gv.T
        best_cos, best_idx = sim.max(1)
        for i, ref_text in enumerate(r):
            j = int(best_idx[i])
            by_layer[layer].append({
                'view_id': row['view_id'],
                'layer': layer,
                'reference': ref_text,
                'generated': g[j],
                'cosine': round(float(best_cos[i]), 4),
                'transcript': row['input'][:TRANSCRIPT_CHARS],
            })

    selected = []
    counts = {}
    for layer in TEXT:
        pairs = sorted(by_layer[layer], key=lambda p: p['cosine'])
        n = len(pairs)
        if n > PER_LAYER:
            step = (n - 1) / (PER_LAYER - 1)
            pairs = [pairs[round(k * step)] for k in range(PER_LAYER)]
        counts[layer] = {'available': n, 'exported': len(pairs)}
        selected.extend(pairs)

    random.Random(SEED).shuffle(selected)
    for i, p in enumerate(selected):
        p['pair_id'] = f'calib-{i:04d}'
    out = BASE / 'calib-pairs.json'
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=1))
    print(json.dumps({'skipped_unparsable': skipped, 'layers': counts, 'total': len(selected), 'path': str(out)}, indent=1))


if __name__ == '__main__':
    main()
