import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.context_candidate_evaluation import evaluate_context_candidate_ranker


class ContextCandidateEvaluationTests(unittest.TestCase):
    def _write_fixture(self, root, same_day=False):
        snapshot_path = root / "snapshots.jsonl"
        label_path = root / "labels.jsonl"
        definitions = {
            "election_campaign_governance": "campaign and elections",
            "economy_trade_jobs": "jobs and prices",
        }
        snapshots, labels = [], []
        for index in range(10):
            event_id = f"event-{index}"
            event_date = f"202{index}-01-02"
            is_election = index % 2 == 0
            snapshots.append({
                "event_id": event_id,
                "event_date": event_date,
                "prediction_cutoff_exclusive": event_date,
                "exploratory_only": True,
                "candidate_sources": [{
                    "title": "Election turnout and ballot update" if is_election else "Jobs prices and wages update",
                    "published_at": event_date if same_day else f"202{index}-01-01",
                    "source_url": f"https://example.test/{index}",
                    "publisher": "Example News",
                }],
            })
            labels.append({
                "event_id": event_id,
                "event_date": event_date,
                "review_status": "draft",
                "family_definitions": definitions,
                "positive_families": ["election_campaign_governance"] if is_election else ["economy_trade_jobs"],
            })
        snapshot_path.write_text("".join(json.dumps(row) + "\n" for row in snapshots), encoding="utf-8")
        label_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
        return snapshot_path, label_path

    def test_evaluator_uses_chronological_split_and_immutable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots, labels = self._write_fixture(root)
            output = root / "report.json"
            report = evaluate_context_candidate_ranker(snapshots, labels, output_path=output)
            self.assertEqual(len(report["split"]["train_event_ids"]), 8)
            self.assertEqual(report["split"]["test_event_ids"], ["event-9"])
            self.assertEqual(report["test_metrics"]["source_evidence_top4"]["events_scored"], 1)
            self.assertEqual(len(report["report_content_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                evaluate_context_candidate_ranker(snapshots, labels, output_path=output)

    def test_evaluator_rejects_same_day_news(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots, labels = self._write_fixture(Path(directory), same_day=True)
            with self.assertRaisesRegex(ValueError, "same-day or later"):
                evaluate_context_candidate_ranker(snapshots, labels)


if __name__ == "__main__":
    unittest.main()
