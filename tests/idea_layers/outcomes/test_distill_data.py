from speechprefix.idea_layer.distill_data import assign_folds, render_input


def test_folds_group_by_event_and_cover_all():
    rows = [{'event_id': f'e{i // 2}'} for i in range(20)]
    assign_folds(rows, 5)
    by_event = {}
    for r in rows:
        by_event.setdefault(r['event_id'], set()).add(r['fold'])
    assert all(len(f) == 1 for f in by_event.values())
    assert {r['fold'] for r in rows} == set(range(5))


def test_render_input_keeps_turn_ids():
    text = render_input({'target_speaker': 'X', 'turns': [{'turn_id': 't0', 'role': 'target', 'text': 'hi'}]})
    assert '[t0] (target) hi' in text
