"""Evaluate held-out generated idea graphs (mentions, ideas, key points, topics, opinions, emotions).

Four checks per view, aggregated over all fold outputs:

1. Quote repair: each evidence exact_quote is kept if it is an exact substring of the cited turn;
   otherwise it is snapped to the best-matching exact substring of that turn when normalized
   SequenceMatcher similarity >= SNAP_MIN, else dropped. Reports exact / snapped / dropped rates.
2. Graph integrity: every link row must resolve to existing ids (idea_mentions, key_point_ideas,
   key_point_mentions, topic_ideas, opinion_mentions, emotion_mentions, opinion_targets) and each key
   point must have exactly one parent idea. Unparsed outputs count as failed.
3. Link scoring: ideas and key points are matched one-to-one (greedy by cosine) at the calibrated
   strict thresholds. Matched key points are checked for attaching to the idea their reference key
   point hangs under (via the idea matching). Matched ideas are checked for citing an evidence turn
   that overlaps a turn cited by the reference mentions of that idea.
4. Multi-scale recall: per layer, a reference item is recalled if some generated item of the same
   layer reaches the calibrated multi-scale threshold (same, broader or narrower).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

TEXT = {'ideas': 'occurrence', 'key_points': 'detail', 'topics': 'label', 'opinions': 'text', 'emotions': 'label'}
ID_KEY = {'mentions': 'mention_id', 'ideas': 'idea_id', 'key_points': 'key_point_id', 'topics': 'topic_id',
          'opinions': 'opinion_id', 'emotions': 'emotion_id'}
# link table -> (source layer, source key, target layer, target key)
LINKS = {
    'idea_mentions': ('ideas', 'idea_id', 'mentions', 'mention_id'),
    'key_point_ideas': ('key_points', 'key_point_id', 'ideas', 'idea_id'),
    'key_point_mentions': ('key_points', 'key_point_id', 'mentions', 'mention_id'),
    'topic_ideas': ('topics', 'topic_id', 'ideas', 'idea_id'),
    'opinion_mentions': ('opinions', 'opinion_id', 'mentions', 'mention_id'),
    'emotion_mentions': ('emotions', 'emotion_id', 'mentions', 'mention_id'),
}
TARGET_LAYER = {'idea': 'ideas', 'key_point': 'key_points'}
SNAP_MIN = 0.9
TURN_RE = re.compile(r'^\[([^\]]+)\] \(([^)]*)\) (.*)$')


def parse_turns(text: str) -> dict[str, str]:
    turns = {}
    for line in text.split('\n'):
        m = TURN_RE.match(line)
        if m:
            turns[m.group(1)] = m.group(3)
    return turns


def norm(s: str) -> str:
    return ' '.join(s.lower().split())


def snap_quote(quote: str, turn_text: str, min_sim: float = SNAP_MIN) -> tuple[str, str | None]:
    """Return (status, text): status is exact, snapped (text = best exact substring) or dropped."""
    if quote and quote in turn_text:
        return 'exact', quote
    if not quote or not turn_text:
        return 'dropped', None
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', turn_text)]
    n = len(quote.split())
    nq = norm(quote)
    best_sim, best_text = 0.0, None
    for width in range(max(1, n - 2), n + 3):
        for i in range(len(spans) - width + 1):
            cand = turn_text[spans[i][0]:spans[i + width - 1][1]]
            sim = difflib.SequenceMatcher(None, nq, norm(cand), autojunk=False).ratio()
            if sim > best_sim:
                best_sim, best_text = sim, cand
    if best_text is not None and best_sim >= min_sim:
        return 'snapped', best_text
    return 'dropped', None


def repair(gen: dict, turns: dict[str, str]) -> tuple[dict, Counter]:
    """Return a copy of gen whose mention evidence keeps only exact or snapped quotes."""
    counts: Counter = Counter()
    mentions = []
    for m in gen.get('mentions', []):
        if not isinstance(m, dict):
            continue
        kept = []
        for e in m.get('evidence', []):
            if not isinstance(e, dict):
                counts['dropped'] += 1
                continue
            tid = e.get('turn_id')
            status, text = snap_quote(e.get('exact_quote') or '', turns.get(tid, ''))
            counts[status] += 1
            if status != 'dropped':
                kept.append({'turn_id': tid, 'exact_quote': text, 'status': status})
        mentions.append({**m, 'evidence': kept})
    return {**gen, 'mentions': mentions}, counts


def _dicts(d: dict, layer: str) -> tuple[list[dict], int]:
    items = d.get(layer, []) or []
    good = [x for x in items if isinstance(x, dict)]
    return good, len(items) - len(good)


def integrity(d: dict) -> dict:
    """Count link rows and key-point parent checks that fail to resolve."""
    ids: dict[str, set] = {}
    violations: Counter = Counter()
    for layer, key in ID_KEY.items():
        items, bad = _dicts(d, layer)
        violations['malformed_item'] += bad
        seen = [x.get(key) for x in items]
        violations['duplicate_id'] += len(seen) - len(set(seen))
        ids[layer] = set(seen)
    checks = broken = 0
    parents: Counter = Counter()
    for table, (src_layer, src_key, dst_layer, dst_key) in LINKS.items():
        rows, bad = _dicts(d, table)
        violations['malformed_row'] += bad
        checks += len(rows) + bad
        broken += bad
        for r in rows:
            ok = r.get(src_key) in ids[src_layer] and r.get(dst_key) in ids[dst_layer]
            if not ok:
                broken += 1
                violations[f'unresolved_{table}'] += 1
            if table == 'key_point_ideas' and r.get(src_key) in ids['key_points']:
                parents[r.get(src_key)] += 1
    rows, bad = _dicts(d, 'opinion_targets')
    violations['malformed_row'] += bad
    checks += len(rows) + bad
    broken += bad
    for r in rows:
        layer = TARGET_LAYER.get(r.get('target_type'))
        ok = layer is not None and r.get('target_id') in ids[layer] and r.get('opinion_id') in ids['opinions']
        if not ok:
            broken += 1
            violations['unresolved_opinion_targets'] += 1
    for kp in ids['key_points']:
        checks += 1
        if parents[kp] != 1:
            broken += 1
            violations['key_point_parent_count_not_1'] += 1
    return {'checks': checks, 'broken': broken, 'violations': {k: v for k, v in violations.items() if v}}


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def greedy_match(sim: list[list[float]], t: float) -> dict[int, int]:
    """One-to-one ref->gen assignment, highest cosine first, only pairs at or above t."""
    cands = sorted(((sim[i][j], i, j) for i in range(len(sim)) for j in range(len(sim[i])) if sim[i][j] >= t),
                   key=lambda c: (-c[0], c[1], c[2]))
    used_r: set[int] = set()
    used_g: set[int] = set()
    out: dict[int, int] = {}
    for _, i, j in cands:
        if i not in used_r and j not in used_g:
            out[i] = j
            used_r.add(i)
            used_g.add(j)
    return out


def _texts(d: dict, layer: str) -> list[tuple[str, str]]:
    key = TEXT[layer]
    return [(x.get(ID_KEY[layer]), x[key]) for x in _dicts(d, layer)[0] if isinstance(x.get(key), str)]


def _parent_map(d: dict) -> dict:
    out: dict = {}
    for r in _dicts(d, 'key_point_ideas')[0]:
        out.setdefault(r.get('key_point_id'), []).append(r.get('idea_id'))
    return out


def _mention_turns(d: dict, idea_id) -> set[str]:
    mids = {r.get('mention_id') for r in _dicts(d, 'idea_mentions')[0] if r.get('idea_id') == idea_id}
    turns = set()
    for m in _dicts(d, 'mentions')[0]:
        if m.get('mention_id') in mids:
            for e in m.get('evidence', []) or []:
                if isinstance(e, dict) and e.get('turn_id'):
                    turns.add(e['turn_id'])
    return turns


def evaluate_view(ref: dict, gen: dict | None, turns: dict[str, str], embed, strict_t: dict, multi_t: dict) -> dict:
    """embed(list[str]) -> list[list[float]]. gen is None when the output is missing or unparsed."""
    gen = gen or {}
    repaired, qcounts = repair(gen, turns)
    integ = integrity(repaired)
    res = {'quotes': dict(qcounts), 'integrity': integ, 'matched': {}, 'parent': Counter(), 'overlap': Counter(),
           'recall': {}}
    # 3. link scoring on one-to-one matched ideas and key points
    maps: dict = {}
    for layer in ('ideas', 'key_points'):
        r_items, g_items = _texts(ref, layer), _texts(repaired, layer)
        vecs = embed([t for _, t in r_items + g_items])
        rv, gv = vecs[:len(r_items)], vecs[len(r_items):]
        sim = [[cos(a, b) for b in gv] for a in rv]
        pairs = greedy_match(sim, strict_t[layer]) if gv and rv else {}
        maps[layer] = {r_items[i][0]: g_items[j][0] for i, j in pairs.items()}
        res['matched'][layer] = {'ref': len(r_items), 'matched': len(maps[layer])}
    idea_map, kp_map = maps['ideas'], maps['key_points']
    ref_parent, gen_parent = _parent_map(ref), _parent_map(repaired)
    for rkp, gkp in kp_map.items():
        rp = ref_parent.get(rkp, [])
        if len(rp) != 1:
            continue
        res['parent']['eligible'] += 1
        gp = gen_parent.get(gkp, [])
        if len(gp) == 1 and idea_map.get(rp[0]) == gp[0]:
            res['parent']['correct'] += 1
        elif len(gp) == 1 and gp[0] not in idea_map.values():
            res['parent']['gen_parent_unmatched'] += 1
        else:
            res['parent']['wrong_parent'] += 1
    for ridea, gidea in idea_map.items():
        rt = _mention_turns(ref, ridea)
        if not rt:
            continue
        res['overlap']['eligible'] += 1
        if rt & _mention_turns(repaired, gidea):
            res['overlap']['overlap'] += 1
    # 4. multi-scale recall per layer
    for layer in TEXT:
        r_items, g_items = _texts(ref, layer), _texts(repaired, layer)
        if not r_items:
            continue
        vecs = embed([t for _, t in r_items + g_items])
        rv, gv = vecs[:len(r_items)], vecs[len(r_items):]
        hit = sum(1 for a in rv if gv and max(cos(a, b) for b in gv) >= multi_t[layer])
        res['recall'][layer] = [hit, len(r_items)]
    return res


def _rate(n: int, d: int):
    return round(n / d, 4) if d else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='fold output directories')
    p.add_argument('--calibration', type=Path, required=True, help='calibration-v1.json')
    p.add_argument('--embedder', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    from speechprefix.idea_layer.distill_eval import embed as _embed
    cal = json.loads(a.calibration.read_text())['chosen_thresholds']
    strict_t, multi_t = cal['strict'], cal['multi']
    memo: dict[str, list[float]] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        todo = sorted({t for t in texts if t not in memo})
        if todo:
            for t, v in zip(todo, _embed(a.embedder, todo).tolist()):
                memo[t] = v
        return [memo[t] for t in texts]

    outputs: dict[str, dict] = {}
    for run in a.runs:
        for path in sorted(run.glob('*.json')):
            d = json.loads(path.read_text())
            if d['view_id'] in outputs:
                raise SystemExit(f'duplicate output for view {d["view_id"]}')
            try:
                raw = json.loads(d['raw']) if isinstance(d['raw'], str) else d['raw']
            except json.JSONDecodeError:
                raw = None
            outputs[d['view_id']] = raw if isinstance(raw, dict) else None
    rows = [json.loads(line) for line in a.pairs.read_text().splitlines() if line.strip()]

    qc: Counter = Counter()
    integ = {'checks': 0, 'broken': 0, 'views_with_violation': 0, 'violations': Counter()}
    parent: Counter = Counter()
    overlap: Counter = Counter()
    matched = {layer: [0, 0] for layer in ('ideas', 'key_points')}
    recall = {layer: [0, 0] for layer in TEXT}
    n_parsed = n_missing = 0
    for row in rows:
        vid = row['view_id']
        if vid not in outputs:
            n_missing += 1
        gen = outputs.get(vid)
        if gen is not None:
            n_parsed += 1
        ref = json.loads(row['target'])
        turns = parse_turns(row['input'])
        r = evaluate_view(ref, gen, turns, embed, strict_t, multi_t)
        if gen is not None:
            qc.update(r['quotes'])
        integ['checks'] += r['integrity']['checks']
        integ['broken'] += r['integrity']['broken']
        integ['violations'].update(r['integrity']['violations'])
        if gen is None or r['integrity']['broken']:
            integ['views_with_violation'] += 1
        parent.update(r['parent'])
        overlap.update(r['overlap'])
        for layer, v in r['matched'].items():
            matched[layer][0] += v['matched']
            matched[layer][1] += v['ref']
        for layer, (h, n) in r['recall'].items():
            recall[layer][0] += h
            recall[layer][1] += n

    n_views = len(rows)
    total_q = sum(qc[k] for k in ('exact', 'snapped', 'dropped'))
    out = {
        'n_views': n_views, 'n_parsed': n_parsed, 'n_unparsed_or_missing': n_views - n_parsed,
        'n_missing_output': n_missing, 'snap_min_similarity': SNAP_MIN, 'embedder': a.embedder,
        'quotes': {'total': total_q, **{k: qc[k] for k in ('exact', 'snapped', 'dropped')},
                   **{f'{k}_rate': _rate(qc[k], total_q) for k in ('exact', 'snapped', 'dropped')}},
        'integrity': {'checks': integ['checks'], 'broken': integ['broken'],
                      'check_failure_rate': _rate(integ['broken'], integ['checks']),
                      'views_with_violation': integ['views_with_violation'],
                      'view_failure_rate': _rate(integ['views_with_violation'], n_views),
                      'violations': dict(integ['violations'])},
        'link_scoring': {
            'strict_thresholds': {k: strict_t[k] for k in ('ideas', 'key_points')},
            'ideas_matched': {'matched': matched['ideas'][0], 'ref': matched['ideas'][1]},
            'key_points_matched': {'matched': matched['key_points'][0], 'ref': matched['key_points'][1]},
            'key_point_parent': {'eligible': parent['eligible'], 'correct': parent['correct'],
                                 'wrong_parent': parent['wrong_parent'],
                                 'gen_parent_unmatched': parent['gen_parent_unmatched'],
                                 'correct_rate': _rate(parent['correct'], parent['eligible'])},
            'idea_evidence_overlap': {'eligible': overlap['eligible'], 'overlap': overlap['overlap'],
                                      'overlap_rate': _rate(overlap['overlap'], overlap['eligible'])},
        },
        'multi_scale_recall': {
            'thresholds': {k: multi_t[k] for k in TEXT},
            'layers': {layer: {'recalled': h, 'ref': n, 'recall': _rate(h, n)} for layer, (h, n) in recall.items()},
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
