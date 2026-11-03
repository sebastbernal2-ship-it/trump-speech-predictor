import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.theme_review import generate_theme_review_packets


class ThemeReviewTests(unittest.TestCase):
    def test_creates_blinded_transcript_only_annotation_copies_and_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots.jsonl"
            events = root / "events.jsonl"
            labels = root / "labels.jsonl"
            taxonomy = {"economy_trade_jobs": "Jobs and prices", "media_information": "News media"}
            snapshot_rows = [{"event_id": f"e{i}", "event_date": f"202{i}-01-01"} for i in range(2)]
            label_rows = [{
                "event_id": f"e{i}", "family_definitions": taxonomy,
                "positive_families": ["media_information"],
            } for i in range(2)]
            event_rows = [{
                "event_id": f"e{i}", "event_date": f"202{i}-01-01", "title": f"secret title {i}",
                "source_url": f"https://example.test/{i}", "target_speaker": "Donald Trump",
                "turns": [
                    {"turn_id": f"e{i}:0", "speaker_name": "Donald Trump", "speaker_role": "target_speaker", "text": f"Target words {i}."},
                    {"turn_id": f"e{i}:1", "speaker_name": "Reporter", "speaker_role": "other_speaker", "text": "Question text."},
                ],
            } for i in range(2)]
            for path, rows in ((snapshots, snapshot_rows), (labels, label_rows), (events, event_rows)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths = [root / "a.jsonl", root / "b.jsonl", root / "key.json"]
            result = generate_theme_review_packets(
                snapshots, events, labels,
                annotator_a_path=paths[0], annotator_b_path=paths[1], key_path=paths[2], seed=7,
                selection_description="two synthetic events for a unit test",
            )
            self.assertEqual(result["item_count"], 2)
            a_rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
            b_rows = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(a_rows, b_rows)
            self.assertNotIn("event_id", a_rows[0])
            self.assertNotIn("secret title", json.dumps(a_rows))
            self.assertNotIn("positive_families", json.dumps(a_rows))
            self.assertEqual(a_rows[0]["taxonomy"][0]["label"], None)
            self.assertIn("First, without consulting the taxonomy", a_rows[0]["instructions"])
            self.assertLess(list(a_rows[0]).index("event_specific_ideas"), list(a_rows[0]).index("taxonomy"))
            self.assertEqual(len(a_rows[0]["target_speaker_turns"]), 1)
            key = json.loads(paths[2].read_text(encoding="utf-8"))
            self.assertEqual(key["schema_version"], 2)
            self.assertEqual(len(key["items"]), 2)
            self.assertEqual(key["selection_description"], "two synthetic events for a unit test")
            self.assertEqual(len(key["selection_manifest_sha256"]), 64)
            self.assertEqual(key["issue_phrase_extractor"]["version"], "compact-issue-phrases-v2")
            with self.assertRaises(FileExistsError):
                generate_theme_review_packets(
                    snapshots, events, labels,
                    annotator_a_path=paths[0], annotator_b_path=root / "b2.jsonl", key_path=root / "k2.json",
                )


if __name__ == "__main__":
    unittest.main()
