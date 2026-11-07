import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from speechprefix.stages.ideas.model import _features, _metrics, evaluate_context_to_idea_classifier, predict_context_ideas


class ContextIssueClassifierTests(unittest.TestCase):
    def test_features_deduplicate_and_strip_publisher_suffix_before_recency_cap(self):
        snapshot = {
            "event_id": "feature-test", "event_date": "2024-01-10",
            "candidate_sources": [
                {"title": "Tariff policy - Reuters", "publisher": "Reuters", "published_at": "2024-01-09"},
                {"title": "Tariff policy - Reuters", "publisher": "Reuters", "published_at": "2024-01-08"},
                {"title": "War and peace - Associated Press", "publisher": "Associated Press", "published_at": "2024-01-09"},
                {"title": "Ukraine energy - Associated Press", "publisher": "Associated Press", "published_at": "2024-01-10"},
            ],
        }
        features = _features(snapshot, headline_limit=1)
        self.assertIn("u:tariff", features)
        self.assertNotIn("u:reuters", features)
        self.assertNotIn("u:war", features)  # date ties preserve captured order
        self.assertNotIn("u:ukraine", features)  # same-day source is excluded

    def test_unseen_target_ideas_are_penalized_in_binary_loss(self):
        metrics = _metrics([], {"new idea"}, all_labels={"known idea"})
        self.assertEqual(metrics["candidate_coverage"], 0.0)
        self.assertEqual(metrics["unseen_target_ideas"], 1.0)
        self.assertGreater(metrics["binary_log_loss"], 10.0)

    def test_context_features_use_only_preceding_week_and_report_immutable_backtest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots.jsonl"
            events = root / "events.jsonl"
            output = root / "report.json"
            start = date(2024, 1, 1)
            snapshot_rows, event_rows = [], []
            for index in range(20):
                day = start + timedelta(days=index)
                is_economy = index % 2 == 0
                snapshot_rows.append({
                    "event_id": f"event-{index}", "event_date": day.isoformat(),
                    "exploratory_only": True,
                    "candidate_sources": [
                        {"title": "Ukraine war and American workers" if is_economy else "Second Amendment and border security",
                         "published_at": (day-timedelta(days=2)).isoformat()},
                        # This older source must not affect the feature vector.
                        {"title": "Iran nuclear talks", "published_at": (day-timedelta(days=15)).isoformat()},
                    ],
                })
                text = "We will end the war in Ukraine and protect American workers." if is_economy else "We must protect the Second Amendment and defend the border."
                event_rows.append({
                    "event_id": f"event-{index}", "event_date": day.isoformat(),
                    "target_speaker": "Donald Trump", "turns": [{
                        "turn_id": f"event-{index}:turn:1", "speaker_name": "Donald Trump",
                        "speaker_role": "target_speaker", "text": text,
                    }],
                })
            snapshots.write_text("".join(json.dumps(row)+"\n" for row in snapshot_rows), encoding="utf-8")
            events.write_text("".join(json.dumps(row)+"\n" for row in event_rows), encoding="utf-8")
            report = evaluate_context_to_idea_classifier(snapshots, events, output_path=output)
            self.assertEqual(report["example_count"], 20)
            self.assertEqual(report["split"]["test_event_ids"], ["event-18", "event-19"])
            self.assertIn("No target transcript", report["features"])
            self.assertIn("binary_log_loss", report["test_metrics"]["headline_conditioned_classifier"])
            self.assertEqual(report["final_fit"]["fit_event_count"], 20)
            self.assertTrue(output.exists())
            prediction = predict_context_ideas(
                {"event_id": "future", "event_date": "2024-01-21", "exploratory_only": True,
                 "candidate_sources": [{"title": "Ukraine war", "published_at": "2024-01-20"}]},
                report,
            )
            self.assertGreater(len(prediction), 0)
            self.assertEqual(prediction[0]["score_type"], "uncalibrated_logit")
            self.assertIn("idea", prediction[0])
            with self.assertRaisesRegex(ValueError, "strictly later"):
                predict_context_ideas(snapshot_rows[-1], report)
            with self.assertRaises(FileExistsError):
                evaluate_context_to_idea_classifier(snapshots, events, output_path=output)

    def test_rejects_same_day_headline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots.jsonl"
            events = root / "events.jsonl"
            day = date(2024, 1, 1)
            rows_s, rows_e = [], []
            for index in range(10):
                current = day + timedelta(days=index)
                rows_s.append({"event_id":str(index),"event_date":current.isoformat(),"exploratory_only":True,
                               "candidate_sources":[{"title":"Ukraine war","published_at":current.isoformat()}]})
                rows_e.append({"event_id":str(index),"event_date":current.isoformat(),"target_speaker":"Donald Trump",
                               "turns":[{"turn_id":f"{index}:turn:1","speaker_name":"Donald Trump","speaker_role":"target_speaker","text":"The Ukraine war must end."}]})
            snapshots.write_text("".join(json.dumps(row)+"\n" for row in rows_s),encoding="utf-8")
            events.write_text("".join(json.dumps(row)+"\n" for row in rows_e),encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same-day/future"):
                evaluate_context_to_idea_classifier(snapshots,events)


if __name__ == "__main__":
    unittest.main()
