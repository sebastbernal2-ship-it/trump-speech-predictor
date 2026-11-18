import pytest

from speechprefix.idea_layer.annotation_review_score import score
from test_annotation_feedback import graph


def test_link_score_distinguishes_missing_link_unmatched_and_unreviewed():
    reference = graph('bridge repairs')
    candidate = graph('bridge repairs', linked=False)
    idea = dict(layer='ideas', reference_id='i', candidate_id='i')
    point = dict(layer='key_points', reference_id='p', candidate_id='p')
    relation = 'key_point_ideas'
    assert score(candidate, reference, [idea])['links'][relation]['unreviewed'] == 1
    assert score(candidate, reference, [idea, {**point, 'candidate_id': None}])['links'][relation]['unmatched_endpoint'] == 1
    assert score(candidate, reference, [idea, point])['links'][relation]['missing_link'] == 1
    assert score(reference, reference, [idea, point])['links'][relation]['matched'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        score(candidate, reference, [idea, idea])
