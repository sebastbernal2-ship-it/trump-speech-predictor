from speechprefix.idea_layer.idea_pair_review import merge_candidates, shortlist


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


def test_pool_deduplicates_label_and_evidence_without_losing_alternatives():
    pool = merge_candidates([('base', graph('a', 'roads', 't')),
                             ('second', graph('a', 'Roads', 't')),
                             ('third', graph('a', 'road repairs', 't'))])
    assert [(i['occurrence'], i['source_run']) for i in pool['ideas']] == [
        ('roads', 'base'), ('road repairs', 'third')]
    assert len({i['idea_id'] for i in pool['ideas']}) == 2
