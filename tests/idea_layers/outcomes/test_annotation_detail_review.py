from speechprefix.idea_layer.annotation_candidates import Candidate
from speechprefix.idea_layer.annotation_detail_review import adjacent_windows


def test_adjacent_windows_preserve_exact_offsets_and_never_cross_turns():
    turns = {'a': 'First.  Second.', 'b': 'Third.'}
    bank = [Candidate('a', 0, 6, 'parsed_sentence'),
            Candidate('a', 8, 15, 'parsed_sentence'),
            Candidate('b', 0, 6, 'parsed_sentence')]
    assert adjacent_windows(bank, turns) == {('a', 0, 15): 'First.  Second.'}
    assert adjacent_windows(bank, turns, max_words=1) == {}
