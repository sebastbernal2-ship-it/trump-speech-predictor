from speechprefix.idea_layer.idea_agreement import align_ideas, subject_signature


def test_subject_aliases_preserve_scope_and_direction():
    assert subject_signature("Ukrainian war vs Russia") == subject_signature("Russian-Ukrainian War")
    assert subject_signature("Russian-Ukrainian War") != subject_signature("Battle of Kyiv")
    assert subject_signature("2022 Russian-Ukrainian War") != subject_signature("2014 Russian-Ukrainian War")
    assert subject_signature("aid from Poland to Ukraine") != subject_signature("aid from Ukraine to Poland")


def test_equivalent_candidates_cannot_double_count_reference():
    a = {"idea_id": "a", "occurrence": "Russian-Ukrainian War"}
    b = {"idea_id": "b", "occurrence": "Ukrainian war vs Russia"}
    result = align_ideas([a, b], [b])
    assert result["matches"] == [{"candidate_id": "a", "reference_id": "b"}]
    assert result["unmatched_candidate_ids"] == ["b"]
