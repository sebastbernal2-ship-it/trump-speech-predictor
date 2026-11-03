import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.theme_review import generate_theme_review_packets
from speechprefix.theme_review_evaluation import evaluate_theme_review_agreement


class ThemeReviewEvaluationTests(unittest.TestCase):
    def _packets(self, root):
        snapshots, events, taxonomy = (root / name for name in ("snapshots.jsonl", "events.jsonl", "taxonomy.jsonl"))
        family_definitions = {"economy_trade_jobs": "Jobs", "media_information": "Media"}
        snapshot_rows, event_rows, taxonomy_rows = [], [], []
        for index in range(2):
            event_id = f"event-{index}"
            snapshot_rows.append({"event_id": event_id, "event_date": f"202{index}-01-01"})
            taxonomy_rows.append({"event_id": event_id, "family_definitions": family_definitions, "positive_families": []})
            event_rows.append({
                "event_id": event_id,
                "event_date": f"202{index}-01-01",
                "target_speaker": "Donald Trump",
                "source_url": f"https://example.test/{index}",
                "turns": [{"turn_id": f"{event_id}:turn:0", "speaker_name": "Donald Trump", "speaker_role": "target_speaker", "text": f"Speech words {index}."}],
            })
        for path, rows in ((snapshots, snapshot_rows), (events, event_rows), (taxonomy, taxonomy_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        a, b, key = (root / name for name in ("a.jsonl", "b.jsonl", "key.json"))
        generate_theme_review_packets(snapshots, events, taxonomy, annotator_a_path=a, annotator_b_path=b, key_path=key)
        a_rows = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
        b_rows = [json.loads(line) for line in b.read_text(encoding="utf-8").splitlines()]
        for index, (left, right) in enumerate(zip(a_rows, b_rows)):
            left["annotation_status"] = right["annotation_status"] = "completed"
            turn_id = left["target_speaker_turns"][0]["turn_id"]
            for family_a, family_b in zip(left["taxonomy"], right["taxonomy"]):
                label = "present" if index == 0 and family_a["family_id"] == "economy_trade_jobs" else "absent"
                family_a["label"] = family_b["label"] = label
                family_a["evidence_turn_ids"] = family_b["evidence_turn_ids"] = [turn_id] if label == "present" else []
            if index == 0:
                right["taxonomy"][1]["label"] = "uncertain"
        for path, rows in ((a, a_rows), (b, b_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return a, b, key

    def test_reports_family_agreement_and_adjudication_without_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, key = self._packets(root)
            output = root / "agreement.json"
            report = evaluate_theme_review_agreement(a, b, key, output_path=output)
            self.assertEqual(report["item_count"], 2)
            self.assertEqual(report["overall_family_agreement"]["observed_agreement"], 0.75)
            self.assertEqual(report["per_family"]["media_information"]["observed_agreement"], 0.5)
            self.assertEqual(len(report["adjudication_required"]), 1)
            self.assertNotIn("Speech words", output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                evaluate_theme_review_agreement(a, b, key, output_path=output)

    def test_rejects_incomplete_or_uncited_positive_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, key = self._packets(root)
            rows = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
            rows[0]["taxonomy"][0]["evidence_turn_ids"] = []
            a.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires supporting turn citations"):
                evaluate_theme_review_agreement(a, b, key)


if __name__ == "__main__":
    unittest.main()
