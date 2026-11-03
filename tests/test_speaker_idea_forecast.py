import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from speechprefix.speaker_idea_forecast import evaluate_speaker_idea_forecast


class SpeakerIdeaForecastTests(unittest.TestCase):
    def _write_events(self, path: Path):
        rows = []
        start = date(2024, 1, 1)
        for index in range(20):
            if index < 14:
                text = "We will end the war in Ukraine."
            else:
                text = "We must protect the Second Amendment."
            rows.append({
                "event_id": f"event-{index}",
                "event_date": (start + timedelta(days=index)).isoformat(),
                "target_speaker": "Donald Trump",
                "turns": [{
                    "turn_id": f"event-{index}:turn:1",
                    "speaker_name": "Donald Trump",
                    "speaker_role": "target_speaker",
                    "text": text,
                }],
            })
        # Same-day outcomes must share a prediction history, not leak into each other.
        rows.append({
            "event_id": "same-day-event",
            "event_date": (start + timedelta(days=19)).isoformat(),
            "target_speaker": "Donald Trump",
            "turns": [{
                "turn_id": "same-day-event:turn:1",
                "speaker_name": "Donald Trump",
                "speaker_role": "target_speaker",
                "text": "Iran war must end.",
            }],
        })
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_walk_forward_selects_recency_model_and_keeps_same_day_events_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            output = root / "evaluation.json"
            self._write_events(events)
            report = evaluate_speaker_idea_forecast(events, output_path=output)
            self.assertEqual(report["split"]["train_event_count"], 16)
            self.assertEqual(report["split"]["validation_event_count"], 2)
            self.assertEqual(report["split"]["test_event_count"], 3)
            delta = report["test_metric_differences"]["recency_weighted_vs_lifetime"]
            self.assertEqual(delta["events"], 3)
            self.assertEqual(delta["date_clusters"], 2)
            self.assertIn("cluster_bootstrap_95pct_ci", delta["ndcg_at_10"])
            self.assertEqual(report["schema_version"], 2)
            errors = report["test_error_analysis_at_10"]
            self.assertEqual(errors["events"], 3)
            self.assertEqual(
                errors["missed_target_phrase_instances"],
                sum(errors["missed_target_decomposition"].values()),
            )
            self.assertGreater(report["test_probability_diagnostic"]["candidate_event_pairs"], 0)
            self.assertIn(report["selected_half_life_days"], (30, 90, 180, 365, 730, None))
            self.assertEqual(len(report["test_predictions"]), 3)
            last_date_predictions = [
                row["top_10_predictions"]
                for row in report["test_predictions"]
                if row["event_date"] == (date(2024, 1, 1) + timedelta(days=19)).isoformat()
            ]
            self.assertEqual(len(last_date_predictions), 2)
            self.assertEqual(
                [item["idea_id"] for item in last_date_predictions[0]],
                [item["idea_id"] for item in last_date_predictions[1]],
            )
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                evaluate_speaker_idea_forecast(events, output_path=output)

    def test_rejects_insufficient_date_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps({
                "event_id": "one",
                "event_date": "2024-01-01",
                "target_speaker": "Donald Trump",
                "turns": [{"turn_id": "one:turn:1", "speaker_name": "Donald Trump",
                           "speaker_role": "target_speaker", "text": "The war in Ukraine must end."}],
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 10 distinct event dates"):
                evaluate_speaker_idea_forecast(path)


if __name__ == "__main__":
    unittest.main()
