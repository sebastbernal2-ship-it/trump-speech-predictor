import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("optional PyTorch dependency is not installed")

from speechprefix.hybrid_train import (
    EVIDENCE_EMPTY_ID,
    PAD_ID,
    QUESTION_EMPTY_ID,
    QuestionAnswerChunks,
    build_vocabulary,
)
from speechprefix.segments import segment_record


class QuestionAnswerDatasetTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "speech_id": "qa-fixture",
            "date": "2025-01-01",
            "event": "Remarks and an Exchange With Reporters",
            "source": "fixture",
            "text": (
                "The President. Earlier transcript context.\n\n"
                "Q. What is the plan?\n\n"
                "The President. We will protect workers and build strong communities "
                "while we create jobs and keep our promises to the American people."
            ),
        }
        self.vocabulary = build_vocabulary([self.record["text"]], min_frequency=1)

    def test_examples_separate_prior_transcript_question_and_answer(self):
        dataset = QuestionAnswerChunks(
            [self.record], self.vocabulary, context_length=8, question_length=8,
            evidence_length=8, target_length=8, stride=8,
        )
        self.assertGreater(len(dataset), 0)
        prefix, question, evidence, target = dataset[0]
        self.assertEqual(tuple(prefix.shape), (8,))
        self.assertEqual(tuple(question.shape), (8,))
        self.assertEqual(tuple(evidence.shape), (8,))
        self.assertEqual(tuple(target.shape), (8,))
        self.assertNotIn(QUESTION_EMPTY_ID, question.tolist())
        self.assertEqual(evidence[0].item(), EVIDENCE_EMPTY_ID)
        self.assertNotEqual(prefix[-1].item(), PAD_ID)

    def test_rejects_same_day_or_later_evidence(self):
        question = next(
            segment for segment in segment_record(self.record)
            if segment["segment_type"] == "question"
        )
        with self.assertRaisesRegex(ValueError, "not known before"):
            QuestionAnswerChunks(
                [self.record], self.vocabulary, context_length=8, question_length=8,
                evidence_length=8, target_length=8, stride=8,
                evidence_by_question={question["segment_id"]: [
                    {"context_id": "same-day", "published_at": "2025-01-01",
                     "title": "same day", "abstract": "not allowed"}
                ]},
            )

    def test_evidence_words_survive_the_transcript_frequency_threshold(self):
        vocabulary = build_vocabulary(
            ["common common"], min_frequency=2, required_tokens=["uniqueevidenceterm"]
        )
        self.assertIn("uniqueevidenceterm", vocabulary)

    def test_rejects_evidence_without_source_attribution(self):
        question = next(
            segment for segment in segment_record(self.record)
            if segment["segment_type"] == "question"
        )
        with self.assertRaisesRegex(ValueError, "no source identifier or URL"):
            QuestionAnswerChunks(
                [self.record], self.vocabulary, context_length=8, question_length=8,
                evidence_length=8, target_length=8, stride=8,
                evidence_by_question={question["segment_id"]: [
                    {"published_at": "2024-12-31", "title": "unattributed", "abstract": "text"}
                ]},
            )

    def test_reporter_floor_cue_does_not_create_qa_training_example(self):
        record = dict(self.record, text="Q. Sir, right here.\n\nThe President. Yes, behind you, please.")
        dataset = QuestionAnswerChunks(
            [record], self.vocabulary, context_length=8, question_length=8,
            evidence_length=8, target_length=8, stride=8,
        )
        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
