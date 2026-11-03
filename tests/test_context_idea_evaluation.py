import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from speechprefix.context_idea_evaluation import evaluate_context_idea_candidates


class ContextIdeaEvaluationTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, same_day_source: bool = False):
        snapshots = root / "snapshots.jsonl"
        records = root / "events.jsonl"
        snapshot_rows, event_rows = [], []
        first = date(2025, 1, 10)
        for index in range(10):
            event_date = first + timedelta(days=index)
            published = event_date if same_day_source and index == 0 else event_date - timedelta(days=1)
            event_id = f"event-{index}"
            snapshot_rows.append({
                "event_id": event_id,
                "event_date": event_date.isoformat(),
                "exploratory_only": True,
                "retrieval_profile": "speaker-topic-families",
                "candidate_sources": [{
                    "title": "Ukraine war escalates",
                    "publisher": "Example News",
                    "published_at": published.isoformat(),
                    "source_url": f"https://example.test/{event_id}",
                }],
            })
            event_rows.append({
                "event_id": event_id,
                "event_date": event_date.isoformat(),
                "target_speaker": "Donald Trump",
                "turns": [{
                    "turn_id": f"{event_id}:turn:1",
                    "speaker_name": "Donald Trump",
                    "speaker_role": "target_speaker",
                    "text": "We must end the war in Ukraine.",
                }],
            })
        snapshots.write_text("".join(json.dumps(row) + "\n" for row in snapshot_rows), encoding="utf-8")
        records.write_text("".join(json.dumps(row) + "\n" for row in event_rows), encoding="utf-8")
        return snapshots, records

    def test_compares_context_and_training_prevalence_on_chronological_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots, records = self._write_inputs(root)
            output = root / "evaluation.json"
            report = evaluate_context_idea_candidates(snapshots, records, output_path=output)
            self.assertEqual(len(report["split"]["train_event_ids"]), 8)
            self.assertEqual(len(report["split"]["validation_event_ids"]), 1)
            self.assertEqual(len(report["split"]["test_event_ids"]), 1)
            self.assertIn("context_evidence_ranker", report["test_metrics"]["top_4"])
            self.assertIn("training_transcript_prevalence", report["test_metrics"]["top_4"])
            self.assertEqual(report["test_predictions"][0]["retrieval_profile"], "speaker-topic-families")
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                evaluate_context_idea_candidates(snapshots, records, output_path=output)

    def test_rejects_same_day_context_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots, records = self._write_inputs(Path(directory), same_day_source=True)
            with self.assertRaisesRegex(ValueError, "same-day or later"):
                evaluate_context_idea_candidates(snapshots, records)


if __name__ == "__main__":
    unittest.main()
