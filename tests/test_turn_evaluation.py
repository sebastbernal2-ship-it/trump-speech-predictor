import unittest

from speechprefix.turn_evaluation import (
    evaluate_exact_wording,
    evaluate_idea_labels,
    evaluate_response_status,
)


class TurnEvaluationTests(unittest.TestCase):
    def test_response_status_is_evaluated_separately(self):
        result = evaluate_response_status(
            ["target_spoke", "other_speaker_spoke"],
            ["target_spoke", "no_target_response"],
        )
        self.assertEqual(result["response_status_accuracy"], 0.5)
        self.assertAlmostEqual(result["response_status_macro_f1"], 1 / 3)
        self.assertEqual(result["response_examples_scored"], 2)

    def test_idea_metrics_score_only_human_labeled_examples(self):
        result = evaluate_idea_labels(
            [{"economy", "tax"}, set(), {"border"}],
            [{"economy", "border"}, set(), {"border"}],
        )
        self.assertEqual(result["idea_examples_scored"], 2)
        self.assertEqual(result["unlabeled_examples"], 1)
        self.assertAlmostEqual(result["idea_micro_precision"], 2 / 3)
        self.assertAlmostEqual(result["idea_micro_recall"], 2 / 3)

    def test_wording_metrics_are_separate_and_punctuation_sensitive_text_match_is_reported(self):
        result = evaluate_exact_wording(
            ["We won!", "The plan changed."],
            ["We won.", "The plan changed."],
        )
        self.assertEqual(result["word_error_rate"], 0.0)
        self.assertEqual(result["exact_word_sequence_match"], 1.0)
        self.assertEqual(result["exact_text_match"], 0.5)

    def test_wording_evaluation_rejects_non_target_empty_references(self):
        with self.assertRaisesRegex(ValueError, "nonempty target-speaker"):
            evaluate_exact_wording(["maybe"], [""])


if __name__ == "__main__":
    unittest.main()
