import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.events import (
    append_event,
    add_prediction_examples,
    parse_timecode,
    parse_timestamped_transcript,
    prediction_example_inputs,
)


PAGE = """Site header
Press gaggle: arrival remarks
StressLens\n2 Topics\n10 Entities\nModeration\n4 Speakers
Full Transcript
Donald Trump 00:00:00-00:00:02 (2 sec)
No Signal (0.034)
We have a crowd.
Question 00:00:02-00:00:03 (1 sec)
No StressLens
Do you plan to reach out?
Donald Trump 00:00:03-00:00:05 (2 sec)
No StressLens
Yeah, I will call her.
Mike Johnson 00:00:05-00:00:06 (1 sec)
No Signal (0)
I sent her a note.
Question 00:00:06-00:00:07 (1 sec)
No StressLens
What did she say?
Mike Johnson 00:00:07-00:00:08 (1 sec)
No Signal (0)
She said thank you.
"""


METADATA = {
    "event_id": "fixture-event",
    "event_date": "2026-11-01",
    "title": "Press gaggle after arrival",
    "source_url": "https://example.test/transcript",
    "target_speaker": "Donald J. Trump",
    "target_speaker_aliases": ["Donald Trump"],
    "page_metadata": {"topics": ["campaign"], "entities": ["Mike Johnson"]},
}


class TimestampedEventTests(unittest.TestCase):
    def test_timecode_parses_minutes_or_hours_and_rejects_invalid_seconds(self):
        self.assertEqual(parse_timecode("01:04"), 64)
        self.assertEqual(parse_timecode("00:01:04"), 64)
        with self.assertRaises(ValueError):
            parse_timecode("00:61:04")

    def test_import_preserves_spans_metadata_and_does_not_guess_links(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        self.assertEqual(event["event_id"], "fixture-event")
        self.assertIn("10 Entities", event["page_metadata"]["raw_header"])
        self.assertEqual(
            [turn["speaker_role"] for turn in event["turns"]],
            ["target_speaker", "questioner", "target_speaker", "other_speaker",
             "questioner", "other_speaker"],
        )
        self.assertEqual(event["turns"][0]["annotations"][0]["score"], 0.034)
        self.assertEqual(event["turns"][1]["annotations"][0]["status"], "unavailable")
        self.assertEqual(event["turns"][0]["text"], "We have a crowd.")
        self.assertEqual(event["turns"][1]["start_timecode"], "00:00:02")
        self.assertEqual(event["prediction_examples"], [])

    def test_prediction_snapshot_excludes_target_and_keeps_other_speaker_outcome_separate(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        question_id = event["turns"][1]["turn_id"]
        trump_id = event["turns"][2]["turn_id"]
        later_question = event["turns"][4]["turn_id"]
        mike_id = event["turns"][5]["turn_id"]
        event = add_prediction_examples(event, [
            {
                "prediction_time_seconds": 3,
                "trigger_turn_ids": [question_id],
                "target_turn_ids": [trump_id],
                "response_status": "target_spoke",
                "response_kind": "direct_answer",
                "idea_labels": ["politics.outreach"],
                "annotation_source": "reviewed-v1",
            },
            {
                "prediction_time_seconds": 7,
                "trigger_turn_ids": [later_question],
                "target_turn_ids": [],
                "outcome_turn_ids": [mike_id],
                "response_status": "other_speaker_spoke",
                "response_kind": "answered_by_other",
                "idea_labels": [],
                "annotation_source": "reviewed-v1",
            },
        ])
        example = event["prediction_examples"][0]
        self.assertEqual(example["visible_turn_ids"], [event["turns"][0]["turn_id"], question_id])
        materialized = prediction_example_inputs(event, example)
        self.assertEqual(materialized["input"]["prefix_text"], "Donald Trump: We have a crowd.")
        self.assertEqual(materialized["input"]["question_text"], "Question: Do you plan to reach out?")
        self.assertEqual(materialized["input"]["trigger_turns"][0]["speaker_role"], "questioner")
        self.assertNotIn("annotations", materialized["input"]["trigger_turns"][0])
        self.assertNotIn("page_metadata", materialized["input"])
        self.assertEqual(materialized["target"]["text"], "Yeah, I will call her.")
        self.assertIn("raw_header", materialized["provenance"]["page_metadata"])
        self.assertEqual(event["prediction_examples"][1]["target_turn_ids"], [])

    def test_model_input_builder_rejects_a_target_leak_in_a_saved_snapshot(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        example = add_prediction_examples(event, [{
            "prediction_time_seconds": 3,
            "trigger_turn_ids": [event["turns"][1]["turn_id"]],
            "target_turn_ids": [event["turns"][2]["turn_id"]],
            "response_status": "target_spoke",
            "annotation_source": "reviewed-v1",
        }])["prediction_examples"][0]
        example["visible_turn_ids"].append(event["turns"][2]["turn_id"])
        with self.assertRaisesRegex(ValueError, "fully observed"):
            prediction_example_inputs(event, example)

    def test_target_speaker_clarifying_turn_can_be_a_prediction_trigger(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        example = add_prediction_examples(event, [{
            "prediction_time_seconds": 3,
            "trigger_turn_ids": [event["turns"][0]["turn_id"], event["turns"][1]["turn_id"]],
            "target_turn_ids": [event["turns"][2]["turn_id"]],
            "response_status": "target_spoke",
            "annotation_source": "reviewed-v1",
        }])["prediction_examples"][0]
        inputs = prediction_example_inputs(event, example)
        self.assertIn("Donald Trump:", inputs["input"]["question_text"])
        self.assertIn("Question:", inputs["input"]["question_text"])

    def test_zero_duration_target_at_cutoff_is_not_visible(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        target = event["turns"][2]
        target["end_seconds"] = target["start_seconds"]
        example = add_prediction_examples(event, [{
            "prediction_time_seconds": 3,
            "trigger_turn_ids": [event["turns"][1]["turn_id"]],
            "target_turn_ids": [target["turn_id"]],
            "response_status": "target_spoke",
            "annotation_source": "reviewed-v1",
        }])["prediction_examples"][0]
        self.assertNotIn(target["turn_id"], example["visible_turn_ids"])

    def test_rejects_target_text_visible_before_prediction_point(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        with self.assertRaisesRegex(ValueError, "at or after"):
            add_prediction_examples(event, [{
                "prediction_time_seconds": 4,
                "trigger_turn_ids": [event["turns"][1]["turn_id"]],
                "target_turn_ids": [event["turns"][2]["turn_id"]],
                "response_status": "target_spoke",
                "annotation_source": "reviewed-v1",
            }])

    def test_jsonl_append_refuses_duplicate_event_ids(self):
        event = parse_timestamped_transcript(PAGE, METADATA)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            append_event(output, event)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["event_id"], event["event_id"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                append_event(output, event)

    def test_source_timestamp_anomalies_are_preserved_but_not_used_for_prediction(self):
        page = PAGE.replace(
            "Donald Trump 00:00:03-00:00:05 (2 sec)",
            "Donald Trump 00:00:01-00:00:05 (4 sec)",
        )
        event = parse_timestamped_transcript(page, METADATA, allow_timestamp_anomalies=True)
        self.assertEqual(event["timestamp_anomalies"][0]["reasons"], ["start_time_out_of_order"])
        with self.assertRaisesRegex(ValueError, "timestamp anomalies"):
            add_prediction_examples(event, [])

    def test_parser_rejects_nonmonotonic_turns(self):
        page = PAGE.replace(
            "Donald Trump 00:00:03-00:00:05 (2 sec)",
            "Donald Trump 00:00:01-00:00:05 (4 sec)",
        )
        with self.assertRaisesRegex(ValueError, "timestamps must be ordered"):
            parse_timestamped_transcript(page, METADATA)


if __name__ == "__main__":
    unittest.main()
