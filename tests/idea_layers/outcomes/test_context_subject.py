import pytest

from speechprefix.idea_layer.classical_annotation import annotate
from speechprefix.idea_layer.context_subject import select_subject


def test_context_supplies_interpretation_but_not_evidence():
    spacy = pytest.importorskip('spacy')
    if not spacy.util.is_package('en_core_web_sm'):
        pytest.skip('local parser unavailable')
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    packet = {'event_id': 'e', 'view_id': 'v', 'turns': [
        {'role': 'context', 'turn_id': 'q', 'text': 'Do you support the relief bill?'},
        {'role': 'target', 'turn_id': 'a', 'text': 'Yes.'},
    ]}
    result = annotate(packet, context_labeler=lambda span, prior: select_subject(span, prior, parser))
    assert result['ideas'][0]['occurrence'] == 'the relief bill'
    assert result['mentions'][0]['evidence'] == [{'turn_id': 'a', 'exact_quote': 'Yes.'}]
    assert select_subject('Yes.', {'role': 'context', 'text': 'Do you support the bill or the amendment?'}, parser) is None
