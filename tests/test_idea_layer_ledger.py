import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from speechprefix.idea_layer.contracts import validate_idea_input_snapshot
from speechprefix.idea_layer.ledger import append_source_records, freeze_idea_input_snapshot, read_source_ledger


class IdeaLayerLedgerTests(unittest.TestCase):
    def _source(self, source_id, kind, content, published, observed, *, status="verified_before_cutoff", **extra):
        row = {
            "schema_version": 1,
            "record_type": "idea-layer-source",
            "source_id": source_id,
            "source_type": kind,
            "source_url": f"https://example.test/{source_id}",
            "publisher": "Example source",
            "published_at": published,
            "observed_at": observed,
            "availability_status": status,
            "availability_evidence": "fixture evidence sha256:" + "0" * 64 if status == "verified_before_cutoff" else "",
            "timestamp_basis": "fixture publisher timestamp field" if status == "verified_before_cutoff" else "",
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "reuse_terms": "Local research; retain source attribution.",
        }
        row.update(extra)
        return row

    def _manifest(self, schedule_id="schedule", fact_source_id="news-oil"):
        return {
            "schema_version": 1,
            "record_type": "idea-layer-event",
            "event_id": "speech-2030-01-20",
            "event_start_at": "2030-01-20T18:00:00Z",
            "prediction_cutoff_at": "2030-01-20T16:00:00Z",
            "event_metadata_source_ids": [schedule_id],
            "known_context": [{
                "fact_id": "fact-oil",
                "fact": "A major oil supply story is in the news.",
                "source_ids": [fact_source_id],
            }],
        }

    def test_append_is_idempotent_and_conflicting_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming, ledger = root / "incoming.jsonl", root / "ledger.jsonl"
            source = self._source(
                "post-1", "trump_post", "We are watching.",
                "2030-01-20T13:00:00Z", "2030-01-20T13:01:00Z",
                account="realDonaldTrump", post_id="123",
            )
            incoming.write_text(json.dumps(source) + "\n", encoding="utf-8")
            self.assertEqual(append_source_records(incoming, ledger)["appended"], 1)
            self.assertEqual(append_source_records(incoming, ledger)["already_present_identically"], 1)
            self.assertEqual(len(read_source_ledger(ledger)), 1)
            source["content"] = "Changed after ingestion."
            source["content_sha256"] = hashlib.sha256(source["content"].encode()).hexdigest()
            incoming.write_text(json.dumps(source) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different content"):
                append_source_records(incoming, ledger)

    def test_freeze_keeps_only_verified_pre_cutoff_sources_and_hashes_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, input_path = root / "event.json", root / "sources.jsonl"
            ledger, output = root / "ledger.jsonl", root / "snapshot.json"
            manifest = self._manifest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            sources = [
                self._source("schedule", "event_metadata", "Official schedule", "2030-01-19T10:00:00Z", "2030-01-19T10:05:00Z"),
                self._source("news-oil", "news", "Oil disruption headline", "2030-01-20T12:00:00Z", "2030-01-20T12:10:00Z", canonical_source_id="news-oil-canonical"),
                self._source("news-oil-later-capture", "news", "Oil disruption headline", "2030-01-20T12:00:00Z", "2030-01-20T14:00:00Z", canonical_source_id="news-oil-canonical"),
                self._source("news-gaza-first", "news", "Gaza ceasefire talks", "2030-01-20T11:00:00Z", "2030-01-20T11:10:00Z", canonical_source_id="news-gaza-canonical"),
                self._source("news-gaza-later", "news", "Gaza ceasefire talks", "2030-01-20T11:00:00Z", "2030-01-20T14:59:00Z", canonical_source_id="news-gaza-canonical"),
                self._source("post-1", "trump_post", "We are watching oil.", "2030-01-20T13:00:00Z", "2030-01-20T13:02:00Z", account="realDonaldTrump", post_id="123"),
                self._source("old", "news", "Old story", "2029-01-01T00:00:00Z", "2029-01-01T00:01:00Z"),
                self._source("history-old", "other_primary_source", "Earlier speech transcript", "2028-01-01T00:00:00Z", "2028-01-02T00:00:00Z", source_role="speaker_history"),
                self._source("archive-old", "other_primary_source", "Old unrelated primary source", "2028-01-01T00:00:00Z", "2028-01-02T00:00:00Z"),
                self._source("unverified", "news", "Unverified story", "2030-01-20T11:00:00Z", "2030-01-20T11:01:00Z", status="unknown"),
                self._source("late", "news", "After-cutoff story", "2030-01-20T16:01:00Z", "2030-01-20T16:02:00Z"),
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
            append_source_records(input_path, ledger)
            result = freeze_idea_input_snapshot(
                manifest_path, ledger, output, lookback_days=30,
                frozen_at=datetime(2030, 1, 20, 15, 0, tzinfo=timezone.utc),
            )
            snapshot = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(snapshot["input_source_ids"]), {"schedule", "news-oil", "news-gaza-later", "post-1", "history-old"})
            self.assertEqual(result["source_count"], 5)
            self.assertTrue(validate_idea_input_snapshot(snapshot)["strict_pre_cutoff"])
            snapshot["lookback_days"] = 31
            with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
                validate_idea_input_snapshot(snapshot)
            with self.assertRaises(FileExistsError):
                freeze_idea_input_snapshot(manifest_path, ledger, output, frozen_at=datetime(2030, 1, 20, 15, 0, tzinfo=timezone.utc))

    def test_freeze_rejects_after_cutoff_and_missing_required_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, source_path, ledger = root / "event.json", root / "sources.jsonl", root / "ledger.jsonl"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            source_path.write_text(json.dumps(self._source(
                "schedule", "event_metadata", "Official schedule", "2030-01-19T10:00:00Z", "2030-01-19T10:05:00Z",
            )) + "\n", encoding="utf-8")
            append_source_records(source_path, ledger)
            with self.assertRaisesRegex(ValueError, "missing from ledger"):
                freeze_idea_input_snapshot(manifest_path, ledger, root / "missing.json", frozen_at=datetime(2030, 1, 20, 15, tzinfo=timezone.utc))
            source_path.write_text(json.dumps(self._source(
                "news-oil", "news", "Oil story", "2030-01-20T12:00:00Z", "2030-01-20T12:01:00Z",
            )) + "\n", encoding="utf-8")
            append_source_records(source_path, ledger)
            with self.assertRaisesRegex(ValueError, "at or after prediction cutoff"):
                freeze_idea_input_snapshot(manifest_path, ledger, root / "late.json", frozen_at=datetime(2030, 1, 20, 16, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
