import json
import unittest
from speechprefix.transcript_processing.semantic_v2 import build, digest


def review(record, segments, **kwargs):
    return {'source_record_sha256': digest(json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))),
            'reviewer': 'development reviewer', 'status': 'reviewed', 'segments': segments, **kwargs}


def seg(t, start, end, role):
    return {'turn_id': t, 'start': start, 'end': end, 'role': role, 'reason': 'development annotation'}


def cite(t, start, end):
    return {'turn_id': t, 'start': start, 'end': end}


def test_context_and_metadata_are_not_target_evidence():
    record = {'turns': [{'turn_id': 'q', 'text': 'Meeting the President of Syria?'},
                        {'turn_id': 'a', 'text': 'Title: Economy\nYes, I think so.'}]}
    parts = [seg('q', 0, 31, 'context'), seg('a', 0, 15, 'metadata'), seg('a', 15, 31, 'target')]
    item = {'id': 'meeting', 'label': 'Expects to meet Syria’s president',
            'evidence': [cite('a', 15, 31)], 'context': [cite('q', 0, 31)]}
    result = build(record, review(record, parts, ideas=[item]))
    assert result['ideas'][0]['evidence'][0]['quote'] == 'Yes, I think so.'
    assert result['ideas'][0]['context'][0]['quote'].startswith('Meeting')
    item['evidence'] = [cite('a', 0, 14)]
    try:
        build(record, review(record, parts, ideas=[item]))
    except ValueError:
        pass
    else:
        raise AssertionError("metadata accepted as speaker evidence")


def test_mapping_and_states():
    record = {'turns': [{'turn_id': 'a', 'text': '  Health\n  wishes.  '}]}
    parts = [seg('a', 0, 20, 'target')]
    item = {'id': 'health', 'label': 'Well wishes for health', 'evidence': [cite('a', 2, 17)]}
    r = review(record, parts, ideas=[item])
    out = build(record, r)
    assert out['segments'][0]['clean_text'] == 'Health wishes.'
    assert out['segments'][0]['clean_to_raw'][6] == [8, 11]
    r.update(status='reviewed_empty', ideas=[])
    assert build(record, r)['status'] == 'reviewed_empty'
    r.update(status='uncertain', reason='incomplete recording')
    assert build(record, r)['status'] == 'uncertain'
    for state in ('unprocessed', 'failed'):
        r.update(status=state, segments=[], reason='not available')
        assert build(record, r)['status'] == state


def test_varied_development_cases():
    # Paraphrased diagnostics, not independent reference annotations.
    cases = [
        ('My best wishes for his recovery.', 'Health', 'Wishes him a recovery'),
        ('The job numbers are excellent and our policy is working.', 'Employment', 'Job numbers are strong and policy is working'),
        ('We mourn the victims of the attacks in Iran.', 'War and terrorism', 'Mourns the victims of attacks in Iran'),
        ('Our policies for investors will not change.', 'Investment policy', 'Policies for investors will remain unchanged'),
        ('The American Pope is a wonderful choice.', 'Religion', 'Praises the American Pope'),
        ('The United Nations has great potential right now.', 'United Nations', 'Says the United Nations has potential'),
        (('Thank you for being here. ' * 400) + 'The factory will create jobs.', 'Employment', 'Factory will create jobs'),
    ]
    for text, topic, idea in cases:
        record = {'turns': [{'turn_id': 't', 'text': text}]}
        evidence = [cite('t', len(text) - len(text.split('.')[-2].lstrip()) - 1, len(text))] if len(text) > 100 else [cite('t', 0, len(text))]
        data = review(record, [seg('t', 0, len(text), 'target')],
                      topics=[{'id': 'topic', 'label': topic, 'evidence': evidence}],
                      ideas=[{'id': 'idea', 'label': idea, 'evidence': evidence}])
        out = build(record, data)
        assert out['ideas'][0]['label'] == idea
        assert out['topics'][0]['evidence'][0]['quote']


if __name__ == '__main__':
    test_context_and_metadata_are_not_target_evidence()
    test_mapping_and_states()
    test_varied_development_cases()
    print('semantic-v2 development checks passed')
