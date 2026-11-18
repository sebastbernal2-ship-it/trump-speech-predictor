from speechprefix.idea_layer.annotation_candidates import Candidate
from speechprefix.idea_layer.annotation_detail_selector import select


def test_parent_conditioning_changes_selection_with_fixed_budget():
    turns = {'z': 'Road repairs begin today.', 'a': 'Schools reopen next week.'}
    bank = [Candidate(t, 0, len(text), 'parsed_sentence') for t, text in turns.items()]
    bank.append(Candidate('a', 0, 7, 'entity'))
    assert select(bank, turns, 'Schools', 1, conditioned=True) == {('a', 0, len(turns['a']))}
    assert select(bank, turns, 'Schools', 1, conditioned=False) == {('z', 0, len(turns['z']))}
