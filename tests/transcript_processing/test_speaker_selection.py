import json

import pytest

from speechprefix.transcript_processing.speaker_selection import digest, select


def sample():
    record = {'turns': [{'turn_id': '1', 'text': 'Hello.'},
                        {'turn_id': '2', 'text': 'Reporter question?'},
                        {'turn_id': '3', 'text': 'Answer.'}]}
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    review = {'source_record_sha256': digest(canonical), 'reviewer': 'reviewer', 'turns': [
        {'turn_id': t['turn_id'], 'text_sha256': digest(t['text']),
         'selection': 'exclude' if t['turn_id'] == '2' else 'include', 'reason': 'reviewed'}
        for t in record['turns']]}
    return record, review


def test_selection_preserves_turn_spans_and_excludes_other_speaker():
    record, review = sample()
    result = select(record, review)
    assert result['text'] == 'Hello.\nAnswer.'
    assert [result['text'][s['start']:s['end']] for s in result['turn_spans']] == ['Hello.', 'Answer.']
    assert result['semantic_status'] == 'not_reviewed'


def test_selection_rejects_stale_or_unresolved_review():
    record, review = sample()
    review['turns'][1]['selection'] = 'unknown'
    with pytest.raises(ValueError, match='unresolved'):
        select(record, review)
    record, review = sample()
    record['turns'][0]['text'] = 'Changed.'
    with pytest.raises(ValueError, match='source record hash'):
        select(record, review)
    record, review = sample()
    review['turns'].pop()
    with pytest.raises(ValueError, match='every source turn'):
        select(record, review)
