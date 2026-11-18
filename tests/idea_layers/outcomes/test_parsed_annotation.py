import pytest

from speechprefix.idea_layer.parsed_annotation import label_span


def test_extracts_noun_phrase_instead_of_discourse_marker():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('local parser model unavailable')
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    assert label_span('Remember, we must protect college sports.', parser) == 'college sports'
    assert label_span('Here we go.', parser) is None
