import tempfile
import unittest
from pathlib import Path

from speechprefix.model import NGramModel, evaluate_continuation, tokenize


class NGramModelTests(unittest.TestCase):
    def test_tokenize_preserves_punctuation_and_normalizes_case(self):
        self.assertEqual(tokenize("Hello, we're here!"), ["hello", ",", "we're", "here", "!"])

    def test_predict_uses_longest_context_and_normalizes(self):
        model = NGramModel.train(["we will win", "we will win", "we will work", "we can win"], order=3)
        predictions = dict(model.predict("we will"))
        self.assertGreater(predictions["win"], predictions["work"])
        self.assertAlmostEqual(sum(predictions.values()), 1.0)

    def test_save_and_load(self):
        model = NGramModel.train(["the people win", "the people work"], order=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = NGramModel.load(path)
        self.assertEqual(model.predict("the people"), loaded.predict("the people"))

    def test_evaluation_holds_out_last_speech(self):
        result = evaluate_continuation(["we will win", "we will work"], order=2)
        self.assertEqual(result["tokens"], 3)
        self.assertGreaterEqual(result["log_loss"], 0)
        self.assertGreaterEqual(result["top1_accuracy"], 0)

    def test_evaluation_requires_speech_level_holdout(self):
        with self.assertRaises(ValueError):
            evaluate_continuation(["only one speech"], order=2)


if __name__ == "__main__":
    unittest.main()
