"""Build supervised (transcript -> annotation graph) pairs and event-grouped folds.

Targets are the frozen first-pass Codex drafts; inputs are transcript packets only.
Folds group by event so no event appears on both sides of a split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .annotation_candidate_eval import load_references

ORDER = ['mentions', 'ideas', 'idea_mentions', 'key_points', 'key_point_ideas', 'key_point_mentions', 'topics', 'topic_ideas',
         'opinions', 'opinion_targets', 'opinion_mentions', 'emotions', 'emotion_mentions']


def render_input(packet: dict) -> str:
    lines = [f"Speaker: {packet['target_speaker']}"]
    for turn in packet['turns']:
        lines.append(f"[{turn['turn_id']}] ({turn['role']}) {turn['text']}")
    return '\n'.join(lines)


def build(root: Path) -> list[dict]:
    raw = (root / 'manifest.json').read_bytes()
    rows = []
    for view, packet, _turns, ref in load_references(root, raw):
        target = {k: ref[k] for k in ORDER if k in ref}
        rows.append({'view_id': view, 'event_id': packet['event_id'],
                     'input': render_input(packet),
                     'target': json.dumps(target, ensure_ascii=False)})
    return rows


def assign_folds(rows: list[dict], k: int = 5) -> None:
    events = sorted({r['event_id'] for r in rows}, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
    fold = {e: i % k for i, e in enumerate(events)}
    for r in rows:
        r['fold'] = fold[r['event_id']]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='frozen review run (manifest, packets, outcomes)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--folds', type=int, default=5)
    a = p.parse_args()
    rows = build(a.root)
    assign_folds(rows, a.folds)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = [sum(r['fold'] == i for r in rows) for i in range(a.folds)]
    print(json.dumps({'views': len(rows), 'events': len({r['event_id'] for r in rows}), 'fold_sizes': counts}))


if __name__ == '__main__':
    main()
