import pytest

from speechprefix.idea_layer.subject_rank_experiment import local_antecedent, proposals, question_subjects


def test_proposals_keep_explicit_result_with_action():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('local parser unavailable')
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    text = 'We strengthened the bridge so that trucks could cross safely.'
    assert ('relation', text) in proposals(next(parser(text).sents), parser)
    assert 'the relief bill' in question_subjects('Do you support the relief bill?', parser)
    assert question_subjects('You support the relief bill.', parser) == []
    text = 'The county repaired the bridge. That helped commuters.'
    first, second = parser(text).sents
    assert local_antecedent(second, first) == text
    assert local_antecedent(second, None) is None
    first, second = parser('The county repaired the bridge. Commuters returned.').sents
    assert local_antecedent(second, first) is None
