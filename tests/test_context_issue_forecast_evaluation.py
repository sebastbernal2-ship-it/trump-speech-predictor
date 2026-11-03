import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from speechprefix.context_issue_forecast import generate_prospective_issue_forecast
from speechprefix.context_issue_forecast_evaluation import evaluate_prospective_issue_forecast
from speechprefix.context_snapshots import seal_prospective_snapshot


class _FixedDateTime(datetime):
    fixed_now = datetime(2025, 5, 14, 16, 30, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed_now if tz is None else cls.fixed_now.astimezone(tz)


class ContextIssueForecastEvaluationTests(unittest.TestCase):
    def test_scores_frozen_forecast_against_outcome_and_prior_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            forecast_path = root / "forecast.json"
            events_path = root / "events.jsonl"
            output_path = root / "evaluation.json"
            snapshot = seal_prospective_snapshot({
                "event_id": "event-target",
                "event_start_at": "2025-05-14T18:00:00Z",
                "prediction_cutoff_at": "2025-05-14T17:00:00Z",
                "event_metadata": {
                    "source_url": "https://schedule.test/event",
                    "published_at": "2025-05-10T09:00:00Z",
                    "observed_at": "2025-05-11T09:00:00Z",
                    "availability_verified_before_cutoff": True,
                },
                "queries": [{
                    "query": '"Donald Trump" Ukraine OR war',
                    "retrieved_at": "2025-05-14T15:00:00Z",
                }],
                "candidate_sources": [{
                    "title": "Ukraine war intensifies before summit",
                    "publisher": "Example News",
                    "source_url": "https://news.test/ukraine",
                    "published_at": "2025-05-13T10:00:00Z",
                    "observed_at": "2025-05-14T15:00:00Z",
                    "query_purposes": ["foreign_policy"],
                }],
            }, frozen_at=datetime(2025, 5, 14, 15, 30, tzinfo=timezone.utc))
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch("speechprefix.context_issue_forecast.datetime", _FixedDateTime):
                generate_prospective_issue_forecast(snapshot_path, forecast_path)

            records = [
                {
                    "event_id": "history-event",
                    "event_date": "2024-05-01",
                    "target_speaker": "Donald Trump",
                    "turns": [{"turn_id": "history:turn:1", "speaker_name": "Donald Trump",
                               "speaker_role": "target_speaker", "text": "We will end the war in Ukraine."}],
                },
                {
                    "event_id": "event-target",
                    "event_date": "2025-05-14",
                    "target_speaker": "Donald Trump",
                    "turns": [{"turn_id": "event:turn:1", "speaker_name": "Donald Trump",
                               "speaker_role": "target_speaker", "text": "We must end the war in Ukraine."}],
                },
            ]
            events_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            report = evaluate_prospective_issue_forecast(
                forecast_path, snapshot_path, events_path, output_path=output_path, top_k=(1, 2),
            )
            self.assertEqual(report["event_id"], "event-target")
            self.assertEqual(report["history_event_count"], 1)
            self.assertEqual(report["metrics"]["top_1"]["saved_context_forecast"]["recall_at_k"], 1.0)
            self.assertTrue(report["saved_context_candidates"][0]["matched_target"])
            self.assertEqual(report["saved_context_candidates"][0]["supporting_turn_ids"], ["event:turn:1"])
            self.assertTrue(output_path.exists())
            with self.assertRaises(FileExistsError):
                evaluate_prospective_issue_forecast(
                    forecast_path, snapshot_path, events_path, output_path=output_path, top_k=(1, 2),
                )

    def test_rejects_forecast_report_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            forecast_path = root / "forecast.json"
            snapshot = seal_prospective_snapshot({
                "event_id": "event-target",
                "event_start_at": "2025-05-14T18:00:00Z",
                "prediction_cutoff_at": "2025-05-14T17:00:00Z",
                "event_metadata": {
                    "source_url": "https://schedule.test/event",
                    "published_at": "2025-05-10T09:00:00Z",
                    "observed_at": "2025-05-11T09:00:00Z",
                    "availability_verified_before_cutoff": True,
                },
                "queries": [{"query": "fixed query", "retrieved_at": "2025-05-14T15:00:00Z"}],
                "candidate_sources": [],
            }, frozen_at=datetime(2025, 5, 14, 15, 30, tzinfo=timezone.utc))
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch("speechprefix.context_issue_forecast.datetime", _FixedDateTime):
                forecast = generate_prospective_issue_forecast(snapshot_path, forecast_path)
            forecast["candidate_count"] = 99
            forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash does not match"):
                evaluate_prospective_issue_forecast(
                    forecast_path, snapshot_path, root / "absent-events.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
