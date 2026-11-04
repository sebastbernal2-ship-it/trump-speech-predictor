import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speechprefix.idea_layer.assistant_review_validation import validate_assistant_review_artifacts


class AssistantReviewValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.blind_id = "blind-item-001"
        self.event_id = "speech-fixture"
        self.text = "We will expand domestic manufacturing."
        self.source_turn_id = f"{self.event_id}:turn:0000"
        self.idea = {
            "idea_id": "manufacturing",
            "idea": "Expand domestic manufacturing.",
            "stance": "support",
            "prominence": "central",
            "confidence": "high",
            "forecastable": True,
            "evidence_turn_ids": [f"{self.blind_id}:turn:0000"],
            "rationale": "The speaker states this directly.",
        }
        self.event = {
            "event_id": self.event_id,
            "event_date": "2030-01-01",
            "source_url": "https://example.test/speech",
        }
        self.turns = [{"turn_id": self.source_turn_id, "text": self.text}]
        self.protocol_hash = "a" * 64
        self._write_fixtures()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _digest(value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _write_json(self, path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _write_jsonl(self, path, rows):
        Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def _write_fixtures(self):
        transcript_hash = hashlib.sha256(
            f"{self.source_turn_id}\t{self.text}".encode()
        ).hexdigest()
        blind_transcript_hash = hashlib.sha256(
            f"{self.blind_id}:turn:0000\t{self.text}".encode()
        ).hexdigest()
        source_record_hash = self._digest(self.event)
        self.key = {
            "schema_version": 2,
            "items": [{
                "review_item_id": self.blind_id,
                "event_id": self.event_id,
                "event_date": self.event["event_date"],
                "source_url": self.event["source_url"],
                "source_record_sha256": source_record_hash,
                "target_transcript_sha256": transcript_hash,
                "target_turn_count": 1,
            }],
        }
        self.key_path = self.root / "key.json"
        self._write_json(self.key_path, self.key)
        self.events_path = self.root / "events.jsonl"
        self._write_jsonl(self.events_path, [self.event])
        self.transcripts_path = self.root / "transcripts.jsonl"
        self._write_jsonl(self.transcripts_path, [{
            "review_item_id": self.blind_id,
            "target_speaker_turns": self.turns,
        }])
        self.pass_a_path = self.root / "pass-a.jsonl"
        pass_a = {
            "schema_version": 1,
            "artifact_type": "assistant-reviewed-outcome-pass-a",
            "blind_item_id": self.blind_id,
            "annotation_policy": "assistant-transcript-only-review-v1",
            "review_status": "assistant-reviewed-transcript-only",
            "coverage_assessment": "full available transcript read",
            "outcome_ideas": [self.idea],
            "review_metadata": {
                "reviewer_type": "AI assistant",
                "provider": "test",
                "model": "test-model",
                "reviewed_at": "2030-01-02T00:00:00Z",
                "protocol_sha256": self.protocol_hash,
                "event_metadata_hidden": True,
                "event_identity_inferable_from_transcript": True,
                "forecasts_news_and_weak_labels_hidden": True,
                "full_transcript_read_before_unblinding": True,
            },
            "target_transcript_blind_sha256": blind_transcript_hash,
        }
        self._write_jsonl(self.pass_a_path, [pass_a])
        pass_a_hash = hashlib.sha256(self.pass_a_path.read_bytes()).hexdigest()
        mapped_idea = dict(self.idea)
        mapped_idea["evidence_turn_ids"] = [self.source_turn_id]
        self.mapped_path = self.root / "mapped.jsonl"
        self.mapped = {
            "schema_version": 1,
            "artifact_type": "assistant-reviewed-transcript-idea-outcomes",
            "annotation_policy": "assistant-transcript-only-review-v1",
            "review_status": "assistant-reviewed-transcript-only",
            "event_id": self.event_id,
            "event_date": self.event["event_date"],
            "source_url": self.event["source_url"],
            "source_record_sha256": source_record_hash,
            "target_transcript_sha256": transcript_hash,
            "review_item_id": self.blind_id,
            "pass_a_blind_artifact_sha256": pass_a_hash,
            "review_key_sha256": hashlib.sha256(self.key_path.read_bytes()).hexdigest(),
            "reviewer": {
                "reviewer_type": "AI assistant",
                "provider": "test",
                "model": "test-model",
                "reviewed_at": "2030-01-02T00:00:01Z",
                "protocol_sha256": self.protocol_hash,
            },
            "unblinded_for_provenance_at": "2030-01-02T00:00:01Z",
            "review_method": {
                "full_target_transcript_reviewed": True,
                "event_metadata_hidden_during_pass_a": True,
                "event_identity_inferable_from_transcript": True,
                "candidate_forecasts_opened": False,
                "forecasts_news_and_weak_labels_opened": False,
                "outcome_labels_frozen_before_unblinding": True,
            },
            "coverage_assessment": "full available transcript read",
            "outcome_ideas": [mapped_idea],
            "review_notes": [],
            "limitations": [],
        }
        self._write_jsonl(self.mapped_path, [self.mapped])

    def _validate(self):
        return validate_assistant_review_artifacts(
            pass_a_paths=[self.pass_a_path],
            mapped_paths=[self.mapped_path],
            key_path=self.key_path,
            events_path=self.events_path,
            transcript_packet_path=self.transcripts_path,
        )

    def test_validates_source_hashes_citations_and_declared_order(self):
        report = self._validate()
        self.assertEqual(report["item_count"], 1)
        self.assertEqual(report["forecastable_outcome_count"], 1)
        self.assertTrue(report["items"][0]["citation_mapping_valid"])
        self.assertEqual(report["items"][0]["pass_a_precedes_unblinding_by_seconds"], 1.0)

    def test_rejects_tampered_source_transcript(self):
        self.turns[0]["text"] += " Extra text."
        self._write_jsonl(self.transcripts_path, [{
            "review_item_id": self.blind_id,
            "target_speaker_turns": self.turns,
        }])
        with self.assertRaisesRegex(ValueError, "source transcript hash"):
            self._validate()

    def test_rejects_changed_mapped_decision(self):
        self.mapped["outcome_ideas"][0]["idea"] = "A different idea."
        self._write_jsonl(self.mapped_path, [self.mapped])
        with self.assertRaisesRegex(ValueError, "differ from frozen Pass A"):
            self._validate()

    def test_rejects_citation_outside_transcript(self):
        self.mapped["outcome_ideas"][0]["evidence_turn_ids"] = ["speech-fixture:turn:9999"]
        self._write_jsonl(self.mapped_path, [self.mapped])
        with self.assertRaisesRegex(ValueError, "unknown source turns"):
            self._validate()

    def test_rejects_forecast_opening_or_bad_review_order(self):
        self.mapped["review_method"]["candidate_forecasts_opened"] = True
        self._write_jsonl(self.mapped_path, [self.mapped])
        with self.assertRaisesRegex(ValueError, "declared unopened"):
            self._validate()
        self._write_fixtures()
        self.mapped["unblinded_for_provenance_at"] = "2030-01-01T00:00:00Z"
        self._write_jsonl(self.mapped_path, [self.mapped])
        with self.assertRaisesRegex(ValueError, "earlier than unblinding"):
            self._validate()


if __name__ == "__main__":
    unittest.main()
