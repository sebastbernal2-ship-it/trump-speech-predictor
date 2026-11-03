import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.forecast import load_records
from speechprefix.rollcall_split import (
    OUTPUT_FILES,
    extract_long_speech_records,
    split_rollcall_events,
)


def event(event_id, record_type, roles):
    return {
        "event_id": event_id,
        "page_metadata": {"factbase_record": {"record_type": record_type}},
        "turns": [{"speaker_role": role} for role in roles],
    }


class RollCallSplitTests(unittest.TestCase):
    def test_creates_disjoint_task_files_without_modifying_master(self):
        rows = [
            event("qa", "Remarks", ["target_speaker", "questioner"]),
            event("speech", "Speech", ["target_speaker"]),
            event("other", "Interview", ["target_speaker"]),
            event("non-target", "Press Briefing", ["questioner", "other_speaker"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "master.jsonl"
            original = "".join(json.dumps(row) + "\n" for row in rows)
            source.write_text(original, encoding="utf-8")
            output = root / "split"

            counts = split_rollcall_events(source, output)

            self.assertEqual(counts, {
                "qa_candidates": 1,
                "speech_candidates": 1,
                "other_trump_events": 1,
                "non_trump_events": 1,
            })
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            actual = {}
            for group, filename in OUTPUT_FILES.items():
                actual[group] = [
                    json.loads(line)["event_id"]
                    for line in (output / filename).read_text(encoding="utf-8").splitlines()
                ]
            self.assertEqual(actual, {
                "qa_candidates": ["qa"],
                "speech_candidates": ["speech"],
                "other_trump_events": ["other"],
                "non_trump_events": ["non-target"],
            })
            with self.assertRaises(FileExistsError):
                split_rollcall_events(source, output)

    def test_long_speech_export_is_target_only_and_legacy_schema_compatible(self):
        def speech(event_id, target_words, other_words=0, questioner=False):
            roles = [
                {"speaker_role": "target_speaker", "text": "trump " * target_words},
                {"speaker_role": "other_speaker", "text": "other " * other_words},
            ]
            if questioner:
                roles.append({"speaker_role": "questioner", "text": "question"})
            return {
                "event_id": event_id,
                "event_date": "2025-01-02",
                "title": f"Speech {event_id}",
                "source_url": f"https://example.test/{event_id}",
                "page_metadata": {"factbase_record": {"record_type": "Speech"}},
                "turns": roles,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech-candidates.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in (
                speech("long", 1000, 250),
                speech("short", 999),
                speech("low-share", 1000, 300),
                speech("has-questioner", 2000, questioner=True),
            )), encoding="utf-8")
            output = root / "long-speeches.jsonl"

            result = extract_long_speech_records(source, output)

            self.assertEqual(result, {"selected_events": 1})
            record = load_records(output)[0]
            self.assertEqual(record["speech_id"], "long")
            self.assertEqual(len(record["text"].split()), 1000)
            self.assertEqual(record["target_word_share"], 0.8)
            self.assertEqual(record["source_record_type"], "Speech")
            with self.assertRaises(FileExistsError):
                extract_long_speech_records(source, output)

    def test_invalid_input_does_not_leave_partial_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "broken.jsonl"
            source.write_text("not json\n", encoding="utf-8")
            output = root / "split"
            with self.assertRaisesRegex(ValueError, "line 1"):
                split_rollcall_events(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".split-*")), [])


if __name__ == "__main__":
    unittest.main()
