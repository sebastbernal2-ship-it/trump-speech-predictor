"""Build speaker-specific text only from explicitly reviewed turn decisions."""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = 'reviewed-turn-selection-v1'


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def select(record, review):
    """Validate a complete review bound to the exact source record and assemble text."""
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    if review.get('source_record_sha256') != digest(canonical):
        raise ValueError('review does not match source record hash')
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('reviewer is required')
    turns = record.get('turns')
    decisions = review.get('turns')
    if not isinstance(turns, list) or not turns or not isinstance(decisions, list) or len(turns) != len(decisions):
        raise ValueError('every source turn requires a decision')
    pieces, spans, seen, cursor = [], [], set(), 0
    for turn, decision in zip(turns, decisions):
        if not isinstance(turn, dict) or not isinstance(decision, dict):
            raise ValueError('turns and decisions must be objects')
        turn_id, text = turn.get('turn_id'), turn.get('text')
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen or not isinstance(text, str):
            raise ValueError('turn IDs must be unique nonempty strings and text must be a string')
        seen.add(turn_id)
        if decision.get('turn_id') != turn_id or decision.get('text_sha256') != digest(text):
            raise ValueError('turn decision does not match source order, ID, and text')
        if decision.get('selection') not in ('include', 'exclude'):
            raise ValueError('unresolved turn decision')
        if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
            raise ValueError('each decision requires a reason')
        if decision['selection'] == 'include':
            if pieces:
                cursor += 1
            start = cursor
            pieces.append(text)
            cursor += len(text)
            spans.append({'turn_id': turn_id, 'start': start, 'end': cursor, 'text_sha256': digest(text)})
    if not pieces:
        raise ValueError('no selected turns; review separately instead of producing empty speech')
    text = '\n'.join(pieces)
    return {'version': VERSION, 'source_record_sha256': digest(canonical),
            'reviewer': review['reviewer'], 'text': text, 'text_sha256': digest(text),
            'turn_spans': spans, 'speaker_selection_status': 'reviewed',
            'semantic_status': 'not_reviewed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='one JSON source record')
    parser.add_argument('review', type=Path, help='one independently prepared selection review')
    parser.add_argument('output', type=Path, help='new output JSON file')
    args = parser.parse_args()
    result = select(json.loads(args.record.read_text(encoding='utf-8')),
                    json.loads(args.review.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=True, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
