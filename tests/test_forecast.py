import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.forecast import evaluate_phrase_forecast, forecast_phrases, load_records, phrase_set


class PhraseForecastTests(unittest.TestCase):
    def records(self):
        texts = [
            "We build strong communities together",
            "We build strong communities today",
            "We build strong communities for everyone",
            "We build strong communities tomorrow",
        ]
        return [
            {"speech_id": f"s{i}", "date": f"202{i}-01-01", "event": "rally", "source": f"source-{i}", "text": text}
            for i, text in enumerate(texts)
        ]

    def test_phrase_extraction_omits_glue_only_phrases(self):
        phrases = phrase_set("we are for the working people")
        self.assertNotIn("we are", phrases)
        self.assertIn("for the working people", phrases)

    def test_phrase_extraction_ignores_editorial_notes_and_abbreviation_tokens(self):
        phrases = phrase_set("Note: The President spoke at 10:00 a.m. U.S. officials attended.")
        self.assertFalse(any("president spoke" in phrase for phrase in phrases))
        self.assertFalse(any(phrase.startswith("u s") for phrase in phrases))

    def test_forecast_ranks_recurring_phrases(self):
        ranked = forecast_phrases(self.records()[:3], top=5)
        self.assertEqual(ranked[0][0], "build strong")
        self.assertAlmostEqual(ranked[0][1], 1.0)

    def test_walk_forward_evaluation(self):
        result = evaluate_phrase_forecast(self.records(), min_history=2)
        self.assertEqual(result["speeches_scored"], 2)
        self.assertGreaterEqual(result["candidate_phrase_brier"], 0)

    def test_loader_requires_metadata_and_sorts_by_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speeches.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in reversed(self.records())), encoding="utf-8")
            loaded = load_records(path)
        self.assertEqual(loaded[0]["speech_id"], "s0")

    def test_loader_rejects_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speeches.jsonl"
            path.write_text(json.dumps({"text": "text"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
