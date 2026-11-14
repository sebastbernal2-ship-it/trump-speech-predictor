import pytest

from speechprefix.idea_layer.classical_annotation import annotate
from speechprefix.idea_layer.encoder_annotation import build_annotator


def test_linked_layers_resolve_to_target_speaker_spans():
    text = 'I support Great Lakes conservation, specifically restoring wetlands. Thank you for protecting wildlife.'
    result = annotate({'event_id': 'e', 'view_id': 'v', 'turns': [
        {'role': 'target', 'turn_id': 't', 'text': text},
        {'role': 'context', 'turn_id': 'c', 'text': 'This must not become evidence.'},
    ]})
    for layer in ('mentions', 'ideas', 'key_points', 'topics', 'opinions', 'emotions'):
        assert result[layer], layer
    assert result['key_points'][0]['detail'] == 'restoring wetlands'
    assert result['opinions'][0]['stance'] == 'supports'
    assert result['emotions'][0]['label'] == 'gratitude'
    assert all(c['turn_id'] == 't' and c['exact_quote'] in text
               for mention in result['mentions'] for c in mention['evidence'])


def test_encoder_cannot_resolve_remote_model_identifier_as_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_annotator(tmp_path / 'missing-model')
