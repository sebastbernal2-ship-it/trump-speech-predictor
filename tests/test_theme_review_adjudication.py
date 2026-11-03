import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.theme_review import generate_theme_review_packets
from speechprefix.theme_review_adjudication import (
    export_adjudicated_theme_labels,
    generate_adjudication_template,
)


class ThemeReviewAdjudicationTests(unittest.TestCase):
    def _fixture(self, root):
        snapshots, events, taxonomy = (root / name for name in ("snapshots.jsonl", "events.jsonl", "taxonomy.jsonl"))
        definitions = {"economy_trade_jobs": "Jobs", "media_information": "News media"}
        snapshot_rows, event_rows, taxonomy_rows = [], [], []
        for index in range(2):
            event_id = f"event-{index}"
            snapshot_rows.append({"event_id": event_id, "event_date": f"202{index}-01-01"})
            taxonomy_rows.append({"event_id": event_id, "family_definitions": definitions, "positive_families": []})
            event_rows.append({
                "event_id": event_id, "event_date": f"202{index}-01-01", "target_speaker": "Donald Trump",
                "source_url": f"https://example.test/{index}",
                "turns": [{"turn_id": f"{event_id}:turn:0", "speaker_name": "Donald Trump", "speaker_role": "target_speaker", "text": f"Speech words {index}."}],
            })
        for path, rows in ((snapshots, snapshot_rows), (events, event_rows), (taxonomy, taxonomy_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        a, b, key = (root / name for name in ("a.jsonl", "b.jsonl", "key.json"))
        generate_theme_review_packets(snapshots, events, taxonomy, annotator_a_path=a, annotator_b_path=b, key_path=key)
        a_rows = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
        b_rows = [json.loads(line) for line in b.read_text(encoding="utf-8").splitlines()]
        key_rows = {item["review_item_id"]: item for item in json.loads(key.read_text(encoding="utf-8"))["items"]}
        decisions = []
        for left, right in zip(a_rows, b_rows):
            identity = key_rows[left["review_item_id"]]
            is_event_zero = identity["event_id"] == "event-0"
            left["annotation_status"] = right["annotation_status"] = "completed"
            turn_id = left["target_speaker_turns"][0]["turn_id"]
            family_decisions = []
            for family_a, family_b in zip(left["taxonomy"], right["taxonomy"]):
                is_present = is_event_zero and family_a["family_id"] == "economy_trade_jobs"
                label_a = "present" if is_present else "absent"
                label_b = "uncertain" if is_event_zero and family_a["family_id"] == "media_information" else label_a
                family_a["label"], family_b["label"] = label_a, label_b
                family_a["evidence_turn_ids"] = [turn_id] if is_present else []
                family_b["evidence_turn_ids"] = [turn_id] if is_present else []
                final_label = "present" if is_present else ("uncertain" if is_event_zero and family_a["family_id"] == "media_information" else "absent")
                family_decisions.append({
                    "family_id": family_a["family_id"],
                    "label": final_label,
                    "evidence_turn_ids": [turn_id] if is_present else [],
                    "adjudication_note": "Reviewed transcript evidence and resolved uncertainty." if label_a != label_b else "",
                })
            decisions.append({
                "review_item_id": identity["review_item_id"],
                "adjudicated_by": "human-reviewer-1",
                "adjudicated_at": "2025-01-02T00:00:00Z",
                "family_decisions": family_decisions,
                "event_specific_ideas": [{
                    "idea": "A cited event-specific idea", "stance": "support", "evidence_turn_ids": [turn_id],
                }] if is_event_zero else [],
            })
        for path, rows in ((a, a_rows), (b, b_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        decision_path = root / "decisions.jsonl"
        decision_path.write_text("".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8")
        return a, b, key, decision_path

    def test_generates_blank_adjudication_template_without_event_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, _, key, _ = self._fixture(root)
            template = root / "template.jsonl"
            result = generate_adjudication_template(a, key, output_path=template)
            rows = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["item_count"], 2)
            self.assertNotIn("event_id", rows[0])
            self.assertTrue(all(decision["label"] is None for decision in rows[0]["family_decisions"]))

    def test_exports_only_explicitly_adjudicated_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, key, decisions = self._fixture(root)
            output = root / "reviewed.jsonl"
            result = export_adjudicated_theme_labels(a, b, key, decisions, output_path=output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            first = next(row for row in rows if row["event_id"] == "event-0")
            self.assertEqual(result["adjudicated_event_count"], 2)
            self.assertEqual(first["review_status"], "human-adjudicated")
            self.assertEqual(first["positive_families"], ["economy_trade_jobs"])
            self.assertEqual(first["uncertain_families"], ["media_information"])
            self.assertEqual(first["family_definitions"]["media_information"], "News media")
            self.assertEqual(len(first["event_specific_ideas"]), 1)
            self.assertNotIn("Speech words", output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                export_adjudicated_theme_labels(a, b, key, decisions, output_path=output)

    def test_requires_a_note_when_adjudicating_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, key, decisions = self._fixture(root)
            decision_rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
            changed = next(row for row in decision_rows if row["family_decisions"][1]["label"] == "uncertain")
            changed["family_decisions"][1]["adjudication_note"] = ""
            decisions.write_text("".join(json.dumps(row) + "\n" for row in decision_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adjudication_note is required"):
                export_adjudicated_theme_labels(a, b, key, decisions, output_path=root / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
