import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from speechprefix.selective_speaker_idea_forecast import (
    evaluate_selective_speaker_idea_forecast,
    forecast_selective_speaker_ideas,
)


class SelectiveSpeakerIdeaForecastTests(unittest.TestCase):
    def _write_events(self, path: Path, count: int = 30) -> None:
        start = date(2024, 1, 1)
        validation_start = date(2024, 10, 11)
        test_start = date(2026, 1, 6)
        rows = []
        for index in range(count):
            if index < 24:
                day = start + timedelta(days=index)
            elif index < 27:
                day = validation_start + timedelta(days=index - 24)
            else:
                day = test_start + timedelta(days=index - 27)
            rows.append({
                "event_id": f"event-{index}",
                "event_date": day.isoformat(),
                "target_speaker": "Donald Trump",
                "turns": [{
                    "turn_id": f"event-{index}:turn:1",
                    "speaker_name": "Donald Trump",
                    "speaker_role": "target_speaker",
                    "text": "The Ukraine war must end.",
                }],
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_forecast_uses_only_strictly_prior_history_and_abstains_below_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            rows = []
            for index, day in enumerate(("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")):
                rows.append({
                    "event_id": f"event-{index}",
                    "event_date": day,
                    "target_speaker": "Donald Trump",
                    "turns": [{
                        "turn_id": f"event-{index}:turn:1",
                        "speaker_name": "Donald Trump",
                        "speaker_role": "target_speaker",
                        "text": "The Ukraine war must end.",
                    }],
                })
            rows[2]["turns"] = None  # same-day content must be skipped before extraction
            rows[3]["turns"] = None  # future content must be skipped before extraction
            events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            forecast = forecast_selective_speaker_ideas(
                events, "2024-01-03", window_days=30, min_history_events=2, max_predictions=1,
            )
            self.assertEqual(len(forecast["predictions"]), 1)
            prediction = forecast["predictions"][0]
            self.assertEqual(prediction["prior_event_support_count"], 2)
            self.assertEqual(
                {row["event_id"] for row in prediction["supporting_events"]},
                {"event-0", "event-1"},
            )
            self.assertTrue(all(row["event_date"] < "2024-01-03" for row in prediction["supporting_events"]))
            abstention = forecast_selective_speaker_ideas(
                events, "2024-01-03", window_days=30, min_history_events=3, max_predictions=1,
            )
            self.assertTrue(abstention["abstained"])
            self.assertEqual(abstention["predictions"], [])

    def test_validation_only_evaluation_selects_from_training_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, report_path = root / "events.jsonl", root / "report.json"
            self._write_events(events)
            rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            rows[-1]["turns"] = None  # reserved-test transcript must not be sent to the extractor
            events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = evaluate_selective_speaker_idea_forecast(
                events, output_path=report_path, target_precision=0.8, minimum_inner_predictions=1,
            )
            self.assertEqual(report["evaluation_type"], "training-selected-selective-speaker-idea-forecast")
            self.assertEqual(report["split"]["validation_event_count"], 3)
            self.assertEqual(report["split"]["reserved_test_start_date_exclusive_for_this_report"], "2026-01-06")
            self.assertEqual(report["scored_event_count_through_locked_validation_end"], 27)
            self.assertNotIn("test_metrics", report)
            self.assertNotIn("test_event_count", report["split"])
            self.assertGreater(report["config_selection"]["feasible_config_count"], 0)
            self.assertTrue(report_path.exists())
            with self.assertRaises(FileExistsError):
                evaluate_selective_speaker_idea_forecast(
                    events, output_path=report_path, target_precision=0.8, minimum_inner_predictions=1,
                )


if __name__ == "__main__":
    unittest.main()
