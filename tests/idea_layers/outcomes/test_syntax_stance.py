import pytest

from speechprefix.idea_layer.annotation_stance_eval import compare
from speechprefix.idea_layer.syntax_stance import classify


def test_stance_evidence_requires_same_act_and_turn():
    assert compare({('requests', 't', 'Vote today.')},
                   {('asserts', 't', 'Vote today.')}, {'t': 'Vote today.'})['overlap'] == 0


def test_local_parser_recovers_imperative_and_obligation():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('local parser unavailable')
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    assert classify('Vote early.', parser) == 'requests'
    assert classify('We have to fix the bridge.', parser) == 'proposes'
    assert classify('Should we fix the bridge?', parser) is None
