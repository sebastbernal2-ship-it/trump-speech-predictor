import pytest

from speechprefix.idea_layer.idea_hierarchy_score import summarize


def test_valid_narrower_idea_is_counted_without_implying_coverage():
    pair = dict(view_id='v', inventory_index=0, reference_key_point_id='p')
    queue = dict(records=[pair, {**pair, 'inventory_index': 1}], verdicts=dict(
        subject_validity=['valid'], scale_relation_to_parent=['narrower'],
        attachment_verdict=['not_warranted']))
    reviewed = dict(**pair, subject_validity='valid', scale_relation_to_parent='narrower',
                    attachment_verdict='not_warranted')
    result = summarize(queue, {'records': [reviewed]})
    assert result['unreviewed'] == 1
    assert result['counts']['subject_validity'] == {'valid': 1}
    assert result['counts']['scale_relation_to_parent'] == {'narrower': 1}
    assert result['counts']['attachment_verdict'] == {'not_warranted': 1}
    with pytest.raises(ValueError, match='duplicate'):
        summarize(queue, {'records': [reviewed, reviewed]})
