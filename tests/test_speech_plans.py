import unittest

from speechprefix.speech_plans import validate_speech_plan


EVENT = {
    "event_id": "event-1",
    "turns": [
        {"turn_id": "turn-1", "speaker_role": "target_speaker"},
        {"turn_id": "turn-2", "speaker_role": "target_speaker"},
        {"turn_id": "turn-3", "speaker_role": "other_speaker"},
    ],
}

PLAN = {
    "schema_version": 1,
    "event_id": "event-1",
    "review_status": "reviewed",
    "reviewed_by": "reviewer",
    "themes": [{
        "id": "theme-1",
        "summary": "Broad theme",
        "turn_ids": ["turn-1", "turn-2"],
        "sections": [{
            "id": "section-1",
            "summary": "Subtopic",
            "turn_ids": ["turn-1", "turn-2"],
            "talking_points": [{
                "id": "point-1",
                "summary": "Specific claim",
                "turn_ids": ["turn-1"],
            }],
        }],
    }],
}


class SpeechPlanValidationTests(unittest.TestCase):
    def test_accepts_reviewed_hierarchy_grounded_in_target_turns(self):
        validate_speech_plan(PLAN, EVENT, require_reviewed=True)

    def test_rejects_draft_as_training_label(self):
        draft = {**PLAN, "review_status": "draft", "reviewed_by": None}
        with self.assertRaisesRegex(ValueError, "only reviewed"):
            validate_speech_plan(draft, EVENT, require_reviewed=True)

    def test_rejects_child_turn_not_covered_by_parent(self):
        plan = {
            **PLAN,
            "themes": [{
                **PLAN["themes"][0],
                "turn_ids": ["turn-1"],
            }],
        }
        with self.assertRaisesRegex(ValueError, "include all turns"):
            validate_speech_plan(plan, EVENT)

    def test_rejects_non_target_speaker_as_semantic_evidence(self):
        plan = {
            **PLAN,
            "themes": [{
                **PLAN["themes"][0],
                "turn_ids": ["turn-1", "turn-2", "turn-3"],
            }],
        }
        with self.assertRaisesRegex(ValueError, "target-speaker turns only"):
            validate_speech_plan(plan, EVENT)

    def test_rejects_unreviewed_or_empty_nodes_in_reviewed_plan(self):
        plan = {
            **PLAN,
            "themes": [{
                **PLAN["themes"][0],
                "sections": [],
            }],
        }
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_speech_plan(plan, EVENT, require_reviewed=True)


if __name__ == "__main__":
    unittest.main()
