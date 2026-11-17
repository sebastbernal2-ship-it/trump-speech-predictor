import hashlib
import json

import pytest

from speechprefix.transcript_processing.layers import process


def test_all_layers_share_clean_text_and_provenance(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'a', 'text': '  Ukraine\n war.  Tax tariff 3.14...🙂'}) + '\n', encoding='utf-8')
    out = tmp_path / 'layers'
    assert process(source, out, 'id', 'text') == 1
    data = (out / 'layers.jsonl').read_bytes()
    row = json.loads(data)
    text = row['clean_text']
    assert [u['text'] for u in row['words']] == text.split()
    assert '.'.join(u['text'] for u in row['sentences']) == text
    assert ''.join(u['text'] for u in row['characters']) == text
    assert all(text[u['start']:u['end']] == u['text'] for layer in ('words', 'sentences', 'characters') for u in row[layer])
    assert any(item['label'] == 'war' for item in row['topics'])
    assert any(item['id'] == 'ukraine war' for item in row['ideas'])
    assert row['semantic_status'] == 'candidate_unreviewed'
    assert row['speaker_attribution'] == 'unverified'
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest['output_sha256'] == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileExistsError):
        process(source, out, 'id', 'text')


def test_failure_and_empty_are_distinct(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text('{"id":"a","text":""}\n{}\n')
    out = tmp_path / 'layers'
    with pytest.raises(ValueError, match='line 2'):
        process(source, out, 'id', 'text')
    assert (out / 'INCOMPLETE').exists()
    assert not (out / 'manifest.json').exists()
    row = json.loads((out / 'layers.jsonl').read_text())
    assert row['semantic_status'] == 'empty_input'
    assert row['topics'] == row['ideas'] == []
