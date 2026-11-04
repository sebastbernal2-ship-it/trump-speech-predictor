import json
from pathlib import Path
import tempfile
import unittest

from speechprefix.idea_layer.retrospective_news_backtest import build_retrospective_news_idea_forecasts


class RetrospectiveNewsBacktestTests(unittest.TestCase):
    def test_uses_only_pre_event_headlines_and_preserves_retrieval_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "snapshots.jsonl"
            output_path = root / "forecast.json"
            row = {
                "event_id": "event-1",
                "event_date": "2025-05-01",
                "retrieved_at": "2026-11-04T00:00:00Z",
                "query_policy": "fixed speaker/topic queries; no target terms",
                "queries": [{"retrieved_at": "2026-11-04T00:00:00Z"}],
                "candidate_sources": [
                    {"title": "Ukraine war peace deal talks continue", "publisher": "A", "published_at": "2025-04-15T10:00:00Z", "source_url": "https://a.example/1"},
                    {"title": "Ukraine war peace deal remains uncertain", "publisher": "B", "published_at": "2025-04-16T10:00:00Z", "source_url": "https://b.example/1"},
                    {"title": "Second Amendment at campaign event", "publisher": "C", "published_at": "2025-05-01T08:00:00Z", "source_url": "https://c.example/1"},
                    {"title": "Trade deal after the speech", "publisher": "D", "published_at": "2025-05-02T08:00:00Z", "source_url": "https://d.example/1"},
                ],
            }
            source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = build_retrospective_news_idea_forecasts(source_path, output_path=output_path, top_k=10)
            forecast = report["forecasts"][0]
            self.assertEqual(forecast["eligible_pre_event_headline_count"], 2)
            self.assertEqual(forecast["excluded_same_day_later_or_invalid_count"], 2)
            predictions = {candidate["idea_id"]: candidate for candidate in forecast["predictions"]}
            self.assertIn("ukraine war", predictions)
            self.assertEqual(
                {source["url"] for source in predictions["ukraine war"]["supporting_sources"]},
                {"https://a.example/1", "https://b.example/1"},
            )
            self.assertTrue(report["evaluation_mode"].startswith("retrospective only"))
            self.assertTrue(any("retrieved after" in item for item in report["limitations"]))
            self.assertTrue(output_path.exists())
            with self.assertRaises(FileExistsError):
                build_retrospective_news_idea_forecasts(source_path, output_path=output_path)

    def test_rejects_duplicate_or_missing_event_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.jsonl"
            row = {"event_id": "dup", "event_date": "2025-01-01", "candidate_sources": []}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique nonempty"):
                build_retrospective_news_idea_forecasts(path)


if __name__ == "__main__":
    unittest.main()
