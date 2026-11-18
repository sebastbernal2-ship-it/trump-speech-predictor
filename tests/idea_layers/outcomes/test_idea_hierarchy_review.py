import pytest

from speechprefix.idea_layer.idea_hierarchy_review import possible_promotions


def test_detail_candidate_is_queued_not_automatically_demoted():
    quote = 'We discussed the Battle of Kyiv.'
    ref = dict(mentions=[dict(mention_id='m', evidence=[dict(turn_id='t', exact_quote=quote)])],
               ideas=[dict(idea_id='i', occurrence='Russia–Ukraine war')],
               key_points=[dict(key_point_id='p', detail='Battle of Kyiv')],
               idea_mentions=[dict(idea_id='i', mention_id='m')],
               key_point_mentions=[dict(key_point_id='p', mention_id='m')],
               key_point_ideas=[dict(key_point_id='p', idea_id='i')])
    span = dict(turn_id='t', start=0, end=len(quote), quote=quote)
    inventory = [dict(label='Battle of Kyiv', evidence=[span])]
    result, = possible_promotions(inventory, ref, {'t': quote})
    assert result['reference_parent_id'] == 'i'
    assert result['exact_detail_label']
    assert result['subject_validity'] is None
    assert result['scale_relation_to_parent'] is None
    assert result['attachment_verdict'] is None
    with pytest.raises(ValueError, match='evidence span'):
        list(possible_promotions(inventory, ref, {'t': 'Different text.'}))
