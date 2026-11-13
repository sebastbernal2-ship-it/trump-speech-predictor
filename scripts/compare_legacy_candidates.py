"""Before/after: classify legacy lexical candidates by whether their evidence lies in reviewed target spans.
Usage: python3 compare_legacy_candidates.py legacy_transcripts.jsonl review.json  (read-only; prints JSON)"""
import json, sys
want = {c['line']: c for c in json.load(open(sys.argv[2]))['cases']}
out = []
with open(sys.argv[1], encoding='utf-8') as f:
    for n, l in enumerate(f, 1):
        if n not in want: continue
        d = json.loads(l); segs = want[n]['review']['segments']
        tgt = [(s['start'], s['end']) for s in segs if s['role'] == 'target']
        ctx = [(s['start'], s['end']) for s in segs if s['role'] == 'context']
        def where(ev):
            a, b = ev['raw_start'], ev['raw_end']
            inn = sum(max(0, min(b, e) - max(a, s)) for s, e in tgt)
            if inn == 0: return 'outside_target'
            return 'in_target' if inn >= (b - a) * 0.9 else 'partly_in_target'
        for kind in ('topics', 'ideas'):
            for it in d[kind]:
                w = [where(e) for e in it['evidence']]
                out.append({'line': n, 'id': d['source_id'], 'kind': kind[:-1], 'label': it['label'], 'evidence_spans': len(w),
                            'in_target': w.count('in_target'), 'partly_in_target': w.count('partly_in_target'), 'outside_target': w.count('outside_target')})
print(json.dumps(out, indent=1))
