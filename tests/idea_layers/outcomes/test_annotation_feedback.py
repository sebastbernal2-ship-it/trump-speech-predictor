from speechprefix.idea_layer.annotation_feedback import diagnostics, link_diagnostics
from speechprefix.idea_layer.transcript_annotations import EDGES, LAYERS


def graph(quote, turn='t', linked=True):
    row = {key: [] for key in (*LAYERS, *EDGES, 'opinion_targets')}
    row['mentions'] = [{'mention_id': 'm', 'evidence': [{'turn_id': turn, 'exact_quote': quote}]}]
    row['ideas'] = [{'idea_id': 'i', 'occurrence': 'road repairs'}]
    row['key_points'] = [{'key_point_id': 'p', 'detail': 'bridge repairs'}]
    row['idea_mentions'] = [{'idea_id': 'i', 'mention_id': 'm'}]
    row['key_point_mentions'] = [{'key_point_id': 'p', 'mention_id': 'm'}]
    if linked:
        row['key_point_ideas'] = [{'idea_id': 'i', 'key_point_id': 'p'}]
    return row


def test_distinguishes_missing_anchor_wrong_span_and_missing_link():
    ref = graph('repairs')
    candidate = graph('bridge repairs', linked=False)
    feedback = diagnostics(candidate, ref, 1)
    assert feedback['ideas']['counts']['overlapping_quote'] == 1
    assert feedback['ideas']['review'][0]['candidates'][0]['verdict'] is None
    assert link_diagnostics(candidate, ref, feedback)['key_point_ideas']['counts'][
        'no_edge_between_shortlisted_endpoints'] == 1
    assert diagnostics(graph('unrelated'), ref, 1)['ideas']['counts']['same_turn_only'] == 1
    assert diagnostics(graph('repairs', turn='other'), ref, 1)['ideas']['counts'][
        'no_candidate_on_cited_turn'] == 1
