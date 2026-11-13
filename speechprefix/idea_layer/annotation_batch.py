"""Prepare transcript-only work packets and collect source-bound annotations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .transcript_annotations import TranscriptAnnotationError, validate_transcript_annotation


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def packet(event: dict) -> dict:
    target = event.get('target_speaker')
    if not isinstance(target, str) or not target.strip() or not isinstance(event.get('turns'), list):
        raise ValueError('missing_target_or_structured_turns')
    turns = []
    seen = set()
    for turn in event['turns']:
        if not isinstance(turn, dict) or not isinstance(turn.get('text'), str):
            raise ValueError('malformed_source_turn')
        text = turn['text']
        if not text.strip():
            continue
        role = turn.get('speaker_role')
        is_target = role == 'target_speaker' or (role is None and str(turn.get('speaker_name', '')).casefold() == target.casefold())
        ident = turn.get('turn_id')
        if is_target and (not isinstance(ident, str) or not ident.strip() or ident in seen):
            raise ValueError('missing_or_duplicate_target_turn_id')
        if is_target:
            seen.add(ident)
        turns.append({'turn_id': ident, 'role': 'target' if is_target else 'context', 'text': text})
    return {'event_id': event['event_id'], 'target_speaker': target, 'turns': turns}


def batch(sources: list[Path], output: Path, submissions: Path | None = None) -> dict:
    """Rebuild packets from raw sources; never overwrite a run or reuse stale evidence."""
    if len({p.resolve() for p in sources}) != len(sources):
        raise ValueError('duplicate source path')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'packets').mkdir()
    (output / 'outcomes').mkdir()
    rows, inputs, missing_sources = [], [], []
    events, views = set(), {}
    expected = set()
    for source in sources:
        try:
            raw = source.read_bytes()
        except FileNotFoundError:
            missing_sources.append(str(source))
            continue
        inputs.append({'path': str(source), 'sha256': digest(raw)})
        for number, line in enumerate(raw.split(b'\n'), 1):
            if not line.strip():
                continue
            row = {'source': str(source), 'line': number, 'source_record_sha256': digest(line)}
            rows.append(row)
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get('event_id'), str) or not event['event_id'].strip():
                    raise ValueError('missing_event_id')
            except (ValueError, UnicodeError):
                row.update(status='blocked', reason='invalid_event_record_or_missing_id')
                continue
            row['event_id'] = event['event_id']
            events.add(event['event_id'])
            try:
                work = packet(event)
            except ValueError as exc:
                row.update(status='blocked', reason=str(exc))
                continue
            targets = {t['turn_id']: t['text'] for t in work['turns'] if t['role'] == 'target'}
            if not targets:
                row.update(status='skipped', reason='no_target_speaker_text')
                continue
            view = digest(json.dumps(work, sort_keys=True, ensure_ascii=False).encode())
            row['view_id'] = view
            if view not in views:
                work['view_id'] = view
                name = f'{view}.json'
                expected.add(name)
                write_json(output / 'packets' / name, work)
                status = {'status': 'pending', 'reason': 'assistant_first_pass_not_submitted'}
                if submissions is not None and (submissions / name).is_file():
                    try:
                        draft = json.loads((submissions / name).read_bytes())
                        if not isinstance(draft, dict) or draft.get('view_id') != view or draft.get('event_id') != work['event_id']:
                            raise TranscriptAnnotationError('submission does not match source view and event')
                        outcome = validate_transcript_annotation(draft, targets)
                        outcome.update(annotation_pass='assistant_first_pass', review_status='pending_user_review',
                                       target_speaker=work['target_speaker'])
                        result = output / 'outcomes' / name
                        write_json(result, outcome)
                        status = {'status': 'annotated', 'outcome_sha256': digest(result.read_bytes())}
                    except (ValueError, UnicodeError) as exc:
                        status = {'status': 'blocked', 'reason': str(exc)}
                views[view] = status
            row.update(views[view])
    manifest = {'schema_version': 1, 'artifact_role': 'outcomes', 'inputs': inputs,
                'missing_sources': missing_sources, 'distinct_event_ids': len(events),
                'source_records': len(rows), 'distinct_applicable_views': len(views),
                'record_status_counts': dict(Counter(r['status'] for r in rows)),
                'view_status_counts': dict(Counter(v['status'] for v in views.values())),
                'unmatched_submissions': sorted(p.name for p in submissions.glob('*.json') if p.name not in expected and p.name != 'generation-report.json') if submissions else [],
                'records': rows}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submissions', type=Path)
    args = parser.parse_args()
    if args.submissions is not None and not args.submissions.is_dir():
        parser.error('--submissions must name an existing directory')
    manifest = batch(args.source, args.output, args.submissions)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'records'}, indent=2))
    if manifest['missing_sources'] or manifest['unmatched_submissions'] or manifest['record_status_counts'].get('blocked'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
