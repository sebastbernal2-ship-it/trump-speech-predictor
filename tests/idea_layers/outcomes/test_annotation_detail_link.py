from speechprefix.idea_layer.annotation_detail_link import rank_parents


def test_detail_overlap_breaks_context_tie_and_irrelevant_labels_abstain():
    ideas = [{'idea_id': 'schools', 'occurrence': 'Schools'},
             {'idea_id': 'roads', 'occurrence': 'Roads'},
             {'idea_id': 'trade', 'occurrence': 'Trade'}]
    ranked = rank_parents('three schools', 'roads and schools were damaged', ideas)
    assert ranked == [(2, 'schools'), (1, 'roads')]
    assert rank_parents('wetlands', 'wildlife habitat', ideas) == []
