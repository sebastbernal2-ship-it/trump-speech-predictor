import json
from pathlib import Path
import tempfile
import unittest

from speechprefix.idea_layer.review_history_forecast import forecast_history_prevalence_for_review_events


class ReviewHistoryForecastTests(unittest.TestCase):
    @staticmethod
    def _jsonl(path, rows):
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_strictly_prior_dates_only_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "history.jsonl"
            targets_path = root / "targets.jsonl"
            output_path = root / "forecast.json"
            rows = []
            for event_id, event_date, text in [
                ("prior-1", "2024-01-01", "We will end the war in Ukraine."),
                ("prior-2", "2024-01-02", "We will end the war in Ukraine."),
                ("target", "2024-01-03", "We must protect the Second Amendment."),
                ("same-day", "2024-01-03", "We must protect the Second Amendment."),
                ("future", "2024-01-04", "We must protect the Second Amendment."),
            ]:
                rows.append({
                    "event_id": event_id,
                    "event_date": event_date,
                    "target_speaker": "Donald Trump",
                    "turns": [{
                        "turn_id": f"{event_id}:turn:1",
                        "speaker_role": "target_speaker",
                        "text": text,
                    }],
                })
            self._jsonl(history_path, rows)
            self._jsonl(targets_path, [{"event_id": "target", "event_date": "2024-01-03"}])
            report = forecast_history_prevalence_for_review_events(
                history_path, targets_path, output_path=output_path,
            )
            forecast = report["forecasts"][0]
            self.assertEqual(forecast["eligible_corpus_record_count"], 2)
            self.assertEqual(forecast["history_events_with_extractable_ideas"], 2)
            self.assertEqual(forecast["latest_eligible_history_date"], "2024-01-02")
            candidate_ids = {item["idea_id"] for item in forecast["predictions"]}
            self.assertIn("ukraine war", candidate_ids)
            self.assertNotIn("amendment protect", candidate_ids)
            for candidate in forecast["predictions"]:
                self.assertTrue(all(
                    item["event_date"] < forecast["event_date"]
                    for item in candidate["supporting_events_sample"]
                ))
            self.assertTrue(output_path.exists())
            with self.assertRaises(FileExistsError):
                forecast_history_prevalence_for_review_events(
                    history_path, targets_path, output_path=output_path,
                )

    def test_target_manifest_rejects_outcome_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.jsonl"
            targets = root / "targets.jsonl"
            self._jsonl(history, [])
            self._jsonl(targets, [{"event_id": "target", "event_date": "2024-01-01", "outcomes": []}])
            with self.assertRaisesRegex(ValueError, "only event_id and event_date"):
                forecast_history_prevalence_for_review_events(history, targets)

    def test_future_malformed_transcript_is_not_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.jsonl"
            targets = root / "targets.jsonl"
            self._jsonl(history, [{
                "event_id": "future",
                "event_date": "2030-01-02",
                "turns": "not parsed for an earlier cutoff",
            }])
            self._jsonl(targets, [{"event_id": "target", "event_date": "2030-01-01"}])
            report = forecast_history_prevalence_for_review_events(history, targets)
            self.assertEqual(report["forecasts"][0]["eligible_corpus_record_count"], 0)
            self.assertEqual(report["forecasts"][0]["predictions"], [])


if __name__ == "__main__":
    unittest.main()
