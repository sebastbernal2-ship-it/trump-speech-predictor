import json

import pytest

from speechprefix.transcript_processing.candidates import clean, extract, process


def test_clean_mapping_and_evidence():
    raw = '  Ukraine\n  war matters.  Taxes and tariffs matter. '
    text, mapping = clean(raw)
    assert text == 'Ukraine war matters. Taxes and tariffs matter.'
    assert all(raw[a:b].isspace() if c == ' ' else raw[a:b] == c
               for c, (a, b) in zip(text, mapping))
    topics, ideas = extract(text, mapping)
    assert any(t['label'] == 'war' for t in topics)
    assert any(i['id'] == 'ukraine war' for i in ideas)
    for item in topics + ideas:
        for ev in item['evidence']:
            assert text[ev['clean_start']:ev['clean_end']] == ev['text']
            assert clean(raw[ev['raw_start']:ev['raw_end']])[0] == ev['text']


def test_process_provenance_and_failure(tmp_path):
    source = tmp_path / 'in.jsonl'
    source.write_text(json.dumps({'id': 'x', 'text': 'Ukraine war.'}) + '\n' + json.dumps({'id': 'x', 'text': ''}) + '\n')
    out = tmp_path / 'out'
    assert process(source, out, 'id', 'text') == 2
    rows = [json.loads(line) for line in (out / 'transcripts.jsonl').read_text().splitlines()]
    assert [r['source_line'] for r in rows] == [1, 2]
    assert rows[1]['status'] == 'empty_input'
    assert json.loads((out / 'manifest.json').read_text())['records'] == 2
    with pytest.raises(FileExistsError):
        process(source, out, 'id', 'text')
    source.write_text('{}\n')
    with pytest.raises(ValueError):
        process(source, tmp_path / 'failed', 'id', 'text')
    assert (tmp_path / 'failed' / 'INCOMPLETE').exists()
