import pytest

from speechprefix.idea_layer.subject_rank_experiment import proposals


def test_proposals_keep_explicit_result_with_action():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('local parser unavailable')
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    text = 'We strengthened the bridge so that trucks could cross safely.'
    assert ('relation', text) in proposals(next(parser(text).sents), parser)
