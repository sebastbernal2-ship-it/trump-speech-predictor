import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.context_snapshots import audit_legacy_snapshots, validate_prospective_snapshot


class ContextSnapshotTests(unittest.TestCase):
    def _valid_snapshot(self):
        return {
            "schema_version": 2,
            "protocol": "prospective-freeze-v2",
            "event_id": "event-1",
            "event_start_at": "2025-05-14T18:00:00Z",
            "prediction_cutoff_at": "2025-05-14T17:00:00Z",
            "snapshot_frozen_at": "2025-05-14T16:00:00Z",
            "event_metadata": {
                "source_url": "https://example.test/event",
                "published_at": "2025-05-10T09:00:00Z",
                "observed_at": "2025-05-11T09:00:00Z",
                "availability_verified_before_cutoff": True,
            },
            "location_metadata": None,
            "queries": [{"query": "predeclared national news query", "retrieved_at": "2025-05-14T15:00:00Z"}],
            "candidate_sources": [{
                "title": "Jobs and prices ahead of election",
                "publisher": "Example News",
                "source_url": "https://example.test/article",
                "published_at": "2025-05-13T10:00:00Z",
                "observed_at": "2025-05-14T15:00:00Z",
            }],
        }

    def test_accepts_timestamped_pre_event_snapshot(self):
        result = validate_prospective_snapshot(self._valid_snapshot())
        self.assertTrue(result["strict_prospective_eligible"])
        self.assertEqual(result["candidate_source_count"], 1)

    def test_rejects_date_only_or_timezone_naive_cutoff(self):
        snapshot = self._valid_snapshot()
        snapshot["prediction_cutoff_at"] = "2025-05-14"
        with self.assertRaisesRegex(ValueError, "RFC 3339 timestamp"):
            validate_prospective_snapshot(snapshot)
        snapshot["prediction_cutoff_at"] = "2025-05-14T17:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            validate_prospective_snapshot(snapshot)

    def test_rejects_retrospective_source_and_unverified_location(self):
        snapshot = self._valid_snapshot()
        snapshot["candidate_sources"][0]["observed_at"] = "2025-05-15T09:00:00Z"
        with self.assertRaisesRegex(ValueError, "after the snapshot was frozen"):
            validate_prospective_snapshot(snapshot)
        snapshot = self._valid_snapshot()
        snapshot["location_metadata"] = {
            "source_url": "https://example.test/location",
            "published_at": "2025-05-10T09:00:00Z",
            "observed_at": "2025-05-11T09:00:00Z",
            "availability_verified_before_cutoff": False,
        }
        with self.assertRaisesRegex(ValueError, "location metadata"):
            validate_prospective_snapshot(snapshot)

    def test_audits_legacy_retrieval_as_exploratory_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "snapshots.jsonl"
            row = {
                "schema_version": 1,
                "event_id": "old-event",
                "event_date": "2024-10-31",
                "context_retrieval_protocol": "metadata-only-v1",
                "retrieved_at": "2026-11-03T13:09:36+00:00",
                "location_hint": "Nevada",
                "location_availability_before_event": "unverified retrospective metadata",
                "candidate_sources": [{
                    "title": "Election report", "date_precision": "day-filtered-from-RSS-pubDate",
                    "published_at": "2024-10-30",
                }],
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "audit.json"
            report = audit_legacy_snapshots(source, output_path=output)
            self.assertEqual(report["strict_prospective_eligible_count"], 0)
            issues = set(report["events"][0]["issues"])
            self.assertIn("snapshot_retrieved_on_or_after_event_date", issues)
            self.assertIn("location_availability_unverified", issues)
            self.assertIn("candidate_publication_times_only_day_precision", issues)
            with self.assertRaises(FileExistsError):
                audit_legacy_snapshots(source, output_path=output)


if __name__ == "__main__":
    unittest.main()
