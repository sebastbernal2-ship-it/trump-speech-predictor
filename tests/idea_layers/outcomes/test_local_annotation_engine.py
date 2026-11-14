from speechprefix.idea_layer.local_annotation_engine import annotate
from speechprefix.idea_layer.local_annotation_evaluate import compare


def test_local_engine_uses_target_text_and_literal_evidence():
    packet = {'event_id': 'new-event', 'view_id': 'view', 'turns': [
        {'role': 'context', 'turn_id': 'c', 'text': 'Questions about Gaza?'},
        {'role': 'target', 'turn_id': 't', 'text': 'I support tariffs.'},
    ]}
    result = annotate(packet)
    assert result == annotate(packet)
    assert [i['occurrence'] for i in result['ideas']] == ['Trade tariffs']
    assert result['mentions'][0]['evidence'] == [{'turn_id': 't', 'exact_quote': 'I support tariffs.'}]
    assert result['opinions'][0]['stance'] == 'supports'
    assert compare(result, result)['opinion_targets']['matched'] == 1
    assert compare(annotate({**packet, 'turns': packet['turns'][:1]}), result)['ideas'] == {
        'matched': 0, 'candidate': 0, 'reference': 1,
    }
