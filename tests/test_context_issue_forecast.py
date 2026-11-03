import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speechprefix.context_issue_forecast import generate_prospective_issue_forecast
from speechprefix.context_snapshots import seal_prospective_snapshot


class ContextIssueForecastTests(unittest.TestCase):
    def _sealed_snapshot(self, now):
        return seal_prospective_snapshot({
            "event_id": "future-event",
            "event_start_at": (now + timedelta(hours=2)).isoformat(),
            "prediction_cutoff_at": (now + timedelta(hours=1)).isoformat(),
            "event_metadata": {
                "source_url": "https://schedule.test/event",
                "published_at": (now - timedelta(hours=2)).isoformat(),
                "observed_at": (now - timedelta(hours=1)).isoformat(),
                "availability_verified_before_cutoff": True,
            },
            "queries": [{
                "query": '"Donald Trump" Iran OR Ukraine',
                "retrieved_at": (now - timedelta(minutes=5)).isoformat(),
            }],
            "candidate_sources": [{
                "title": "Ukraine war intensifies before summit",
                "publisher": "Example News",
                "source_url": "https://news.test/ukraine",
                "published_at": (now - timedelta(days=1)).isoformat(),
                "observed_at": (now - timedelta(minutes=5)).isoformat(),
                "query_purposes": ["foreign_policy"],
            }],
        }, frozen_at=now)

    def test_generates_immutable_cited_issue_candidates_before_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime.now(timezone.utc)
            snapshot_path = root / "snapshot.json"
            output = root / "forecast.json"
            snapshot_path.write_text(json.dumps(self._sealed_snapshot(now)), encoding="utf-8")
            report = generate_prospective_issue_forecast(snapshot_path, output)
            idea = next(item for item in report["candidates"] if item["idea_id"] == "ukraine war")
            self.assertEqual(idea["idea"], "Ukraine war")
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["candidate_extractor"]["version"], "compact-issue-phrases-v2")
            self.assertEqual(len(report["candidate_extractor"]["lexicon_sha256"]), 64)
            self.assertEqual(idea["supporting_sources"][0]["query_purposes"], ["foreign_policy"])
            self.assertLess(report["forecast_generated_at"], report["prediction_cutoff_at"])
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["snapshot_sha256"], report["snapshot_sha256"])
            self.assertNotIn("target_ideas", saved)
            with self.assertRaises(FileExistsError):
                generate_prospective_issue_forecast(snapshot_path, output)

    def test_requires_a_sealed_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime.now(timezone.utc)
            snapshot = self._sealed_snapshot(now)
            snapshot.pop("snapshot_sha256")
            path = root / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed snapshot hash"):
                generate_prospective_issue_forecast(path, root / "forecast.json")


if __name__ == "__main__":
    unittest.main()
