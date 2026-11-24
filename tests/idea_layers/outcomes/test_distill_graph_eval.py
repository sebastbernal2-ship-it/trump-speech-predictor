from speechprefix.idea_layer.distill_graph_eval import (
    evaluate_view, greedy_match, integrity, parse_turns, repair, snap_quote)


def test_snap_quote_exact_snapped_dropped():
    turn = 'We will work with Prime Minister Abe on trying to get them back.'
    assert snap_quote('Prime Minister Abe', turn) == ('exact', 'Prime Minister Abe')
    status, text = snap_quote('we will work with prime minister abe on trying', turn)
    assert status == 'snapped' and text in turn
    assert snap_quote('completely unrelated words here', turn) == ('dropped', None)
    assert snap_quote('Abe', '') == ('dropped', None)


def test_repair_drops_unknown_turn_and_keeps_counts():
    turns = {'t0': 'Hello there my friends.'}
    gen = {'mentions': [{'mention_id': 'm1', 'evidence': [
        {'turn_id': 't0', 'exact_quote': 'Hello there'},
        {'turn_id': 't9', 'exact_quote': 'Hello there'},
        {'turn_id': 't0', 'exact_quote': 'Hello ther my friends'},
    ]}]}
    out, counts = repair(gen, turns)
    assert counts == {'exact': 1, 'snapped': 1, 'dropped': 1}
    assert [e['turn_id'] for e in out['mentions'][0]['evidence']] == ['t0', 't0']


def test_parse_turns_reads_turn_lines_only():
    text = 'Speaker: X\n[e:turn:0000] (target) First line.\n[e:turn:0001] (context) Second.'
    assert parse_turns(text) == {'e:turn:0000': 'First line.', 'e:turn:0001': 'Second.'}


def _graph(**kw):
    base = {'mentions': [{'mention_id': 'm1', 'evidence': []}], 'ideas': [{'idea_id': 'i1', 'occurrence': 'a'}],
            'key_points': [{'key_point_id': 'k1', 'detail': 'd'}], 'idea_mentions': [], 'key_point_ideas': [],
            'key_point_mentions': [], 'topics': [], 'topic_ideas': [], 'opinions': [], 'opinion_targets': [],
            'opinion_mentions': [], 'emotions': [], 'emotion_mentions': []}
    base.update(kw)
    return base


def test_integrity_flags_unresolved_and_parent_count():
    ok = _graph(idea_mentions=[{'idea_id': 'i1', 'mention_id': 'm1'}],
                key_point_ideas=[{'key_point_id': 'k1', 'idea_id': 'i1'}])
    assert integrity(ok)['broken'] == 0
    bad = _graph(idea_mentions=[{'idea_id': 'i9', 'mention_id': 'm1'}])
    r = integrity(bad)
    assert r['broken'] == 2  # unresolved idea link + key point with no parent
    assert r['violations']['unresolved_idea_mentions'] == 1
    assert r['violations']['key_point_parent_count_not_1'] == 1
    two = _graph(key_point_ideas=[{'key_point_id': 'k1', 'idea_id': 'i1'}] * 2)
    assert integrity(two)['violations']['key_point_parent_count_not_1'] == 1


def test_integrity_opinion_targets_by_type():
    g = _graph(opinions=[{'opinion_id': 'o1', 'text': 't'}],
               opinion_targets=[{'opinion_id': 'o1', 'target_type': 'key_point', 'target_id': 'k1'},
                                {'opinion_id': 'o1', 'target_type': 'bogus', 'target_id': 'k1'}],
               key_point_ideas=[{'key_point_id': 'k1', 'idea_id': 'i1'}])
    r = integrity(g)
    assert r['violations'] == {'unresolved_opinion_targets': 1}


def test_greedy_match_is_one_to_one():
    sim = [[0.9, 0.8], [0.85, 0.2]]
    assert greedy_match(sim, 0.5) == {0: 0}
    assert greedy_match(sim, 0.95) == {}


def _fake_embed(texts):
    # identical text -> identical unit vector; otherwise orthogonal axes per distinct text
    vocab = {}
    out = []
    for t in texts:
        vocab.setdefault(t, len(vocab))
        v = [0.0] * 16
        v[vocab[t] % 16] = 1.0
        out.append(v)
    return out


def test_evaluate_view_end_to_end_with_fake_embedder():
    turns = {'t0': 'Abductions were a disgrace. Working with Abe.'}
    ref = _graph(
        mentions=[{'mention_id': 'm1', 'evidence': [{'turn_id': 't0', 'exact_quote': 'a disgrace'}]}],
        ideas=[{'idea_id': 'i1', 'occurrence': 'Abductions'}, {'idea_id': 'i2', 'occurrence': 'Working'}],
        key_points=[{'key_point_id': 'k1', 'detail': 'Victims'}],
        idea_mentions=[{'idea_id': 'i1', 'mention_id': 'm1'}],
        key_point_ideas=[{'key_point_id': 'k1', 'idea_id': 'i1'}],
    )
    gen = _graph(
        mentions=[{'mention_id': 'm1', 'evidence': [{'turn_id': 't0', 'exact_quote': 'a disgrace.'}]}],
        ideas=[{'idea_id': 'g1', 'occurrence': 'Abductions'}],
        key_points=[{'key_point_id': 'gk', 'detail': 'Victims'}],
        idea_mentions=[{'idea_id': 'g1', 'mention_id': 'm1'}],
        key_point_ideas=[{'key_point_id': 'gk', 'idea_id': 'g1'}],
    )
    strict = {'ideas': 0.5, 'key_points': 0.5}
    multi = {k: 0.5 for k in ('ideas', 'key_points', 'topics', 'opinions', 'emotions')}
    r = evaluate_view(ref, gen, turns, _fake_embed, strict, multi)
    assert r['matched']['ideas'] == {'ref': 2, 'matched': 1}
    assert r['parent'] == {'eligible': 1, 'correct': 1}
    assert r['overlap'] == {'eligible': 1, 'overlap': 1}
    assert r['recall']['ideas'] == [1, 2]
    assert r['quotes'] == {'exact': 1}
    missing = evaluate_view(ref, None, turns, _fake_embed, strict, multi)
    assert missing['matched']['ideas']['matched'] == 0
