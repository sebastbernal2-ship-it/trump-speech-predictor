import unittest

from speechprefix.segments import segment_record


class SegmentTests(unittest.TestCase):
    def test_explicit_question_and_president_answer_are_linked(self):
        record = {
            "speech_id": "fixture",
            "date": "2025-01-01",
            "event": "Remarks and an Exchange With Reporters",
            "source": "fixture",
            "text": (
                "The President. Good morning.\n\n"
                "Q. What is the plan?\n\n"
                "The President. We will act today.\n\n"
                "Mr. Reporter. Thank you.\n\n"
                "The President. My prepared closing."
            ),
        }
        segments = segment_record(record)
        self.assertEqual(
            [item["segment_type"] for item in segments],
            ["presidential_remarks", "question", "presidential_answer", "other_speaker_turn", "presidential_remarks"],
        )
        self.assertEqual(segments[2]["response_to"], segments[1]["segment_id"])
        self.assertIsNone(segments[4]["response_to"])

    def test_unlabeled_question_mark_is_not_promoted_to_question(self):
        record = {
            "speech_id": "fixture", "date": "2025-01-01", "event": "Remarks",
            "source": "fixture", "text": "Isn't this a good day?\n\nWe agree.",
        }
        segments = segment_record(record)
        self.assertEqual([item["segment_type"] for item in segments], ["unattributed_text", "unattributed_text"])
        self.assertEqual([item["label_confidence"] for item in segments], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
