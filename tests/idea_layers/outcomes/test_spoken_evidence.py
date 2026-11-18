from speechprefix.idea_layer.annotation_candidates import propose
from speechprefix.idea_layer.classical_annotation import annotate
from speechprefix.idea_layer.spoken_evidence import has_spoken_evidence


def test_excludes_directions_without_stripping_mixed_speech():
    assert not has_spoken_evidence('[Silence is observed for the victims of 9/11.]')
    assert not has_spoken_evidence('[Pause] [Applause].')
    assert has_spoken_evidence('[Applause] We support the bill.')


def test_direction_only_turns_do_not_create_annotation_nodes():
    packet = {'event_id': 'e', 'view_id': 'v', 'turns': [
        {'role': 'target', 'turn_id': 't', 'text': '[Silence is observed for the victims of 9/11.]'},
        {'role': 'target', 'turn_id': 's', 'text': '[Applause] We support the bill.'},
    ]}
    assert {c.turn_id for c in propose(packet)} == {'s'}
    result = annotate(packet)
    assert [m['evidence'] for m in result['mentions']] == [
        [{'turn_id': 's', 'exact_quote': '[Applause] We support the bill.'}]
    ]
