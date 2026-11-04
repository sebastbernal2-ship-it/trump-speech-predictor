import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speechprefix.idea_layer.contracts import validate_pre_speech_forecast, validate_idea_input_snapshot
from speechprefix.cli import main as cli_main
from speechprefix.idea_layer.ranker import (
    make_idea_proposal_set,
    rank_pre_speech_ideas,
    validate_idea_proposal_set,
)


class EvidenceWeightedRankerTests(unittest.TestCase):
    def _snapshot(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        cutoff = now + timedelta(hours=3)
        start = now + timedelta(hours=4)
        manifest = {
            "schema_version": 1,
            "record_type": "idea-layer-event",
            "event_id": "synthetic-event-001",
            "event_start_at": start.isoformat(),
            "prediction_cutoff_at": cutoff.isoformat(),
            "event_metadata_source_ids": ["synthetic-schedule"],
            "known_context": [],
        }

        def source(source_id, kind, content, published, observed, **extra):
            return {
                "schema_version": 1,
                "record_type": "idea-layer-source",
                "source_id": source_id,
                "canonical_source_id": f"canonical:{source_id}",
                "source_type": kind,
                "source_url": f"https://synthetic.invalid/{source_id}",
                "publisher": "Synthetic Publisher",
                "published_at": published.isoformat(),
                "observed_at": observed.isoformat(),
                "availability_status": "verified_before_cutoff",
                "availability_evidence": "synthetic fixture evidence",
                "timestamp_basis": "synthetic fixture timestamp",
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reuse_terms": "Synthetic fixture only",
                **extra,
            }

        sources = [
            source("synthetic-schedule", "event_metadata", "Synthetic event schedule", now - timedelta(days=2), now - timedelta(days=2)),
            source("synthetic-news", "news", "Synthetic news headline about energy security", now - timedelta(hours=4), now - timedelta(hours=3)),
            source(
                "synthetic-post", "trump_post", "Synthetic post about energy security",
                now - timedelta(hours=2), now - timedelta(hours=1), account="synthetic-account", post_id="synthetic-post-1",
            ),
        ]
        snapshot = {
            "schema_version": 1,
            "record_type": "idea-layer-input-snapshot",
            "event_id": manifest["event_id"],
            "prediction_cutoff_at": cutoff.isoformat(),
            "snapshot_created_at": now.isoformat(),
            "lookback_days": 30,
            "event_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "event_manifest": manifest,
            "input_source_ids": sorted(row["source_id"] for row in sources),
            "sources": sources,
        }
        snapshot["snapshot_sha256"] = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        validate_idea_input_snapshot(snapshot)
        return snapshot, now

    def _proposal_set(self, snapshot, now):
        return make_idea_proposal_set(
            snapshot,
            [
                {"idea_id": "idea:energy", "idea": "Energy security", "evidence_source_ids": ["synthetic-post"]},
                {"idea_id": "idea:news", "idea": "Regional energy developments", "evidence_source_ids": ["synthetic-news"]},
                {"idea_id": "idea:event", "idea": "Event logistics", "evidence_source_ids": ["synthetic-schedule"]},
            ],
            generator="synthetic-fixture-human-proposals",
            proposal_set_id="synthetic-proposal-set-001",
            generated_at=now,
        )

    def test_ranks_only_cited_proposals_and_emits_a_sealed_forecast(self):
        snapshot, now = self._snapshot()
        proposals = self._proposal_set(snapshot, now)
        forecast = rank_pre_speech_ideas(snapshot, proposals)
        self.assertEqual(forecast["candidates"][0]["idea_id"], "idea:energy")
        self.assertEqual(forecast["candidates"][0]["score_kind"], "ranking_score")
        self.assertGreater(forecast["candidates"][0]["score"], forecast["candidates"][1]["score"])
        self.assertEqual(forecast["proposal_set_sha256"], proposals["proposal_set_sha256"])
        self.assertEqual(forecast["input_snapshot_sha256"], snapshot["snapshot_sha256"])
        self.assertEqual(validate_pre_speech_forecast(forecast, snapshot["event_manifest"], snapshot["sources"])["candidate_count"], 3)

    def test_proposal_set_is_snapshot_bound_and_integrity_checked(self):
        snapshot, now = self._snapshot()
        proposals = self._proposal_set(snapshot, now)
        self.assertEqual(validate_idea_proposal_set(proposals, snapshot)["proposal_count"], 3)
        tampered = json.loads(json.dumps(proposals))
        tampered["generator"] = "changed"
        with self.assertRaisesRegex(ValueError, "sha256 does not match"):
            validate_idea_proposal_set(tampered, snapshot)
        wrong_source = [{"idea_id": "bad", "idea": "Unsupported", "evidence_source_ids": ["not-in-snapshot"]}]
        with self.assertRaisesRegex(ValueError, "outside its input snapshot"):
            make_idea_proposal_set(snapshot, wrong_source, generator="synthetic-test", generated_at=now)
        leaked = [{"idea_id": "bad", "idea": "Unsupported", "evidence_source_ids": ["synthetic-news"], "target_ideas": ["leak"]}]
        with self.assertRaisesRegex(ValueError, "exactly idea_id"):
            make_idea_proposal_set(snapshot, leaked, generator="synthetic-test", generated_at=now)
        cutoff = datetime.fromisoformat(snapshot["prediction_cutoff_at"])
        with self.assertRaisesRegex(ValueError, "strictly before"):
            make_idea_proposal_set(snapshot, [], generator="synthetic-test", generated_at=cutoff)
        with self.assertRaisesRegex(ValueError, "cannot predate"):
            make_idea_proposal_set(snapshot, [], generator="synthetic-test", generated_at=now - timedelta(seconds=1))

    def test_proposal_and_ranking_cli_round_trip_with_synthetic_snapshot(self):
        snapshot, _ = self._snapshot()
        proposals = [
            {"idea_id": "idea:energy", "idea": "Energy security", "evidence_source_ids": ["synthetic-post"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            raw_proposals = root / "proposals.json"
            proposal_set_path = root / "proposal-set.json"
            forecast_path = root / "forecast.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            raw_proposals.write_text(json.dumps(proposals), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main([
                    "idea-proposals-seal", "--snapshot", str(snapshot_path), "--input", str(raw_proposals),
                    "--generator", "synthetic-test", "--output", str(proposal_set_path),
                ]), 0)
                self.assertEqual(cli_main([
                    "idea-rank", "--snapshot", str(snapshot_path), "--proposals", str(proposal_set_path),
                    "--output", str(forecast_path),
                ]), 0)
            forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
            self.assertEqual(forecast["candidates"][0]["idea_id"], "idea:energy")
            self.assertEqual(forecast["candidates"][0]["score_kind"], "ranking_score")

    def test_forecast_content_tampering_breaks_hash(self):
        snapshot, now = self._snapshot()
        forecast = rank_pre_speech_ideas(snapshot, self._proposal_set(snapshot, now))
        forecast["candidates"][0]["score"] += 1
        with self.assertRaisesRegex(ValueError, "forecast_sha256 does not match"):
            validate_pre_speech_forecast(forecast, snapshot["event_manifest"], snapshot["sources"])


if __name__ == "__main__":
    unittest.main()
