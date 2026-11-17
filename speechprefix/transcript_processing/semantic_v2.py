"""Review-driven, source-offset semantic targets; no lexical labels are inferred."""
import hashlib
import json

from .candidates import clean

VERSION = 'semantic-targets-v2'


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def build(record, review):
    """Bind explicit segment and semantic review to an immutable source record.

    Evidence offsets are Unicode code points in the original turn, half-open.
    The caller supplies interpretations; this function checks provenance, not meaning.
    """
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    source_hash = digest(canonical)
    if review.get('source_record_sha256') != source_hash:
        raise ValueError('source record hash mismatch')
    status = review.get('status')
    if status not in {'unprocessed', 'failed', 'uncertain', 'reviewed', 'reviewed_empty'}:
        raise ValueError('invalid status')
    if status in {'failed', 'uncertain'} and not str(review.get('reason', '')).strip():
        raise ValueError('failure or uncertainty requires a reason')
    turns = record.get('turns')
    if not isinstance(turns, list) or not turns:
        raise ValueError('source turns required')
    by_id = {}
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get('turn_id'), str) or not turn['turn_id'] or turn['turn_id'] in by_id or not isinstance(turn.get('text'), str):
            raise ValueError('invalid source turn')
        by_id[turn['turn_id']] = turn['text']
    result = {'version': VERSION, 'source_record_sha256': source_hash,
              'status': status, 'reason': review.get('reason'), 'topics': [], 'ideas': [],
              'segments': [], 'semantic_validation': 'not_independently_validated'}
    if status in {'unprocessed', 'failed'}:
        if review.get('segments') or review.get('topics') or review.get('ideas'):
            raise ValueError('unprocessed or failed records cannot contain outputs')
        return result
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('reviewer required')
    result['reviewer'] = review['reviewer']
    segments = review.get('segments')
    if not isinstance(segments, list):
        raise ValueError('segment review required')
    # A complete partition makes every excluded character explicit, including metadata.
    for turn_id, raw in by_id.items():
        entries = [s for s in segments if isinstance(s, dict) and s.get('turn_id') == turn_id]
        cursor = 0
        for s in entries:
            start, end = s.get('start'), s.get('end')
            if type(start) is not int or type(end) is not int or start != cursor or end <= start or end > len(raw) or s.get('role') not in {'target', 'context', 'metadata', 'uncertain'}:
                raise ValueError('segments must partition each turn in order')
            if not isinstance(s.get('reason'), str) or not s['reason'].strip():
                raise ValueError('segment reason required')
            text, mapping = clean(raw[start:end])
            result['segments'].append({**s, 'raw_sha256': digest(raw[start:end]), 'clean_text': text,
                                       'clean_to_raw': [[a + start, b + start] for a, b in mapping]})
            cursor = end
        if cursor != len(raw):
            raise ValueError('segments must cover every turn')
    if len(result['segments']) != len(segments):
        raise ValueError('unknown segment turn')
    if status in {'reviewed', 'reviewed_empty'} and any(s['role'] == 'uncertain' for s in result['segments']):
        raise ValueError('unresolved segments require uncertain status')
    def citation(c, allowed):
        if not isinstance(c, dict):
            raise ValueError('invalid citation')
        turn_id, start, end = c.get('turn_id'), c.get('start'), c.get('end')
        if turn_id not in by_id or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(by_id[turn_id]):
            raise ValueError('invalid citation offsets')
        if not any(s['turn_id'] == turn_id and s['role'] in allowed and s['start'] <= start < end <= s['end'] for s in result['segments']):
            raise ValueError('citation outside permitted segment')
        return {'turn_id': turn_id, 'start': start, 'end': end, 'quote': by_id[turn_id][start:end]}
    for kind in ('topics', 'ideas'):
        items = review.get(kind, [])
        if not isinstance(items, list):
            raise ValueError('targets must be lists')
        ids = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in ids or not isinstance(item.get('label'), str) or not item['label'].strip():
                raise ValueError('target requires unique ID and label')
            ids.add(item['id'])
            evidence = item.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('target speaker evidence required')
            context = item.get('context', [])
            if not isinstance(context, list):
                raise ValueError('context must be a list')
            result[kind].append({'id': item['id'], 'label': item['label'],
                                 'evidence': [citation(c, {'target'}) for c in evidence],
                                 'context': [citation(c, {'context', 'target'}) for c in context]})
    if status == 'reviewed_empty' and not any(s['role'] == 'target' and s['clean_text'] for s in result['segments']):
        raise ValueError('reviewed empty requires nonempty target speech')
    if status == 'reviewed_empty' and (result['topics'] or result['ideas']):
        raise ValueError('reviewed empty cannot contain targets')
    if status == 'reviewed' and not (result['topics'] or result['ideas']):
        raise ValueError('use reviewed_empty for an explicit empty inventory')
    return result
