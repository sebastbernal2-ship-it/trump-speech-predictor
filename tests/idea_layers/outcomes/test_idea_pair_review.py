from speechprefix.idea_layer.idea_pair_review import shortlist


def graph(identifier, label, turn):
    return {'ideas': [{'idea_id': identifier, 'occurrence': label}],
            'mentions': [{'mention_id': 'm', 'evidence': [{'turn_id': turn, 'exact_quote': 'text'}]}],
            'idea_mentions': [{'idea_id': identifier, 'mention_id': 'm'}]}


def test_shared_turn_is_review_signal_not_equivalence():
    rows = list(shortlist(graph('a', 'roads', 't'), graph('b', 'schools', 't'), 1))
    assert rows[0]['pairs'][0]['shared_turn']
    assert rows[0]['pairs'][0]['verdict'] is None


def test_aligned_aliases_are_not_queued():
    assert list(shortlist(graph('a', 'Ukrainian war vs Russia', 't'),
                          graph('b', 'Russian-Ukrainian War', 't'), 1)) == []
