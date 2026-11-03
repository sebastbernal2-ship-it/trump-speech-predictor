import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.context_theme_evaluation import evaluate_context_theme_ranker


class ContextThemeEvaluationTests(unittest.TestCase):
    def _write_fixture(self, root, *, same_day=False, differing_definitions=False):
        snapshots_path = root / "snapshots.jsonl"
        labels_path = root / "labels.jsonl"
        categories = {"election": "campaigns", "economy": "jobs and prices"}
        snapshots, labels = [], []
        for index in range(10):
            event_id = f"event-{index}"
            day = f"202{index}-01-02"
            election = index % 2 == 0
            snapshots.append({
                "event_id": event_id,
                "event_date": day,
                "prediction_cutoff_exclusive": day,
                "exploratory_only": True,
                "event_title_metadata_only": "must not enter features",
                "candidate_sources": [{
                    "title": "ballot campaign vote" if election else "jobs prices manufacturing",
                    "published_at": day if same_day else f"202{index}-01-01",
                    "source_url": f"https://example.test/{index}",
                }],
            })
            labels.append({
                "event_id": event_id,
                "event_date": day,
                "review_status": "draft",
                "family_definitions": categories if not (differing_definitions and index == 9) else {"election": "campaigns"},
                "positive_families": ["election"] if election else ["economy"],
            })
        snapshots_path.write_text("".join(json.dumps(row) + "\n" for row in snapshots), encoding="utf-8")
        labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
        return snapshots_path, labels_path

    def test_evaluator_splits_chronologically_and_saves_immutable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots_path, labels_path = self._write_fixture(root)
            output_path = root / "report.json"
            report = evaluate_context_theme_ranker(snapshots_path, labels_path, output_path=output_path)
            self.assertEqual(len(report["split"]["train_events"]), 8)
            self.assertEqual(len(report["split"]["validation_events"]), 1)
            self.assertEqual(report["split"]["test_events"], ["event-9"])
            self.assertEqual(len(report["report_content_sha256"]), 64)
            self.assertTrue(output_path.exists())
            with self.assertRaises(FileExistsError):
                evaluate_context_theme_ranker(snapshots_path, labels_path, output_path=output_path)

    def test_evaluator_rejects_same_day_context(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots_path, labels_path = self._write_fixture(Path(directory), same_day=True)
            with self.assertRaisesRegex(ValueError, "non-pre-event"):
                evaluate_context_theme_ranker(snapshots_path, labels_path)

    def test_evaluator_rejects_inconsistent_theme_taxonomy(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots_path, labels_path = self._write_fixture(Path(directory), differing_definitions=True)
            with self.assertRaisesRegex(ValueError, "definitions differ"):
                evaluate_context_theme_ranker(snapshots_path, labels_path)


if __name__ == "__main__":
    unittest.main()
