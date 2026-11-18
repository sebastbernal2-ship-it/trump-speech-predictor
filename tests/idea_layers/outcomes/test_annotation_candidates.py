import pytest

from speechprefix.idea_layer.annotation_candidate_eval import score
from speechprefix.idea_layer.annotation_candidates import propose


def test_reference_overlap_stays_in_same_turn_and_checks_repeated_quotes():
    candidates = {('t', 8, 13), ('other', 0, 4)}
    counts = score(candidates, {'t': 'quote;  quote rest'}, {('t', 'quote'), ('t', 'rest')})
    assert counts == {'candidates': 2, 'reference': 2, 'exact': 1, 'overlap_50': 1}


def test_parser_proposes_parent_scoped_clauses_and_arguments():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('optional local parser weights not installed')
    text = '  We supported the relief bill because the storm damaged three schools.'
    packet = {'turns': [dict(turn_id='t', role='target', text=text),
                        dict(turn_id='c', role='context', text='What happened?')]}
    candidates = propose(packet, spacy.load('en_core_web_sm'))
    assert all(c.turn_id == 't' and text[c.start:c.end].strip() == text[c.start:c.end]
               for c in candidates)
    assert any(c.kind == 'clause' and 'storm damaged' in text[c.start:c.end] for c in candidates)
    assert any(c.kind == 'argument' and 'three schools' == text[c.start:c.end] for c in candidates)
    assert all(c.parent_start <= c.start < c.end <= c.parent_end
               for c in candidates if c.parent_start is not None)
