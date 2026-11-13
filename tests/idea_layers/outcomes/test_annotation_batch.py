import json

import pytest

from speechprefix.idea_layer.annotation_batch import batch
from speechprefix.idea_layer.annotation_generate import generate
from test_transcript_annotations import sample


def test_batch_source_binding_and_deduplication(tmp_path, monkeypatch):
    event = {'event_id': 'event-1', 'target_speaker': 'Trump', 'turns': [
        {'turn_id': 't1', 'speaker_role': 'target_speaker', 'text': 'We support this.'},
        {'turn_id': 't2', 'speaker_role': 'interviewer', 'text': 'We support this.'},
    ]}
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(map(json.dumps, [event, event, {**event, 'event_id': 'empty', 'turns': []}])))
    intake = tmp_path / 'intake'
    manifest = batch([source], intake)
    assert manifest['record_status_counts'] == {'pending': 2, 'skipped': 1}
    assert manifest['distinct_applicable_views'] == 1
    view = manifest['records'][0]['view_id']
    packet = intake / 'packets' / f'{view}.json'
    submissions = tmp_path / 'submissions'
    submissions.mkdir()
    draft = {**sample(), 'view_id': view}

    def fake_run(command, **kwargs):
        from subprocess import CompletedProcess
        assert '--no-tools' in command
        return CompletedProcess(command, 0, stdout=json.dumps(draft))

    monkeypatch.setattr('speechprefix.idea_layer.annotation_generate.subprocess.run', fake_run)
    assert generate(packet, submissions, provider='fake', model='fake', timeout=1)['status'] == 'generated'
    review = tmp_path / 'review'
    result = batch([source], review, submissions)
    assert result['record_status_counts'] == {'annotated': 2, 'skipped': 1}
    outcome = json.loads((review / 'outcomes' / packet.name).read_text())
    assert outcome['artifact_role'] == 'outcomes'
    assert outcome['review_status'] == 'pending_user_review'
    with pytest.raises(FileExistsError):
        batch([source], review)
    draft['mentions'][0]['evidence'][0]['turn_id'] = 't2'
    (submissions / packet.name).write_text(json.dumps(draft))
    failed = batch([source], tmp_path / 'rejected', submissions)
    assert failed['view_status_counts'] == {'blocked': 1}
