import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from speechprefix.idea_layer.contracts import validate_idea_input_snapshot
from speechprefix.idea_layer.headline_proposals import GENERATOR_VERSION, propose_headline_issue_phrases
from speechprefix.idea_layer.ranker import rank_pre_speech_ideas, validate_idea_proposal_set
from speechprefix.cli import main as cli_main


class HeadlineProposalTests(unittest.TestCase):
    def _snapshot(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        cutoff = now + timedelta(hours=3)
        start = now + timedelta(hours=4)
        manifest = {
            "schema_version": 1,
            "record_type": "idea-layer-event",
            "event_id": "synthetic-news-event",
            "event_start_at": start.isoformat(),
            "prediction_cutoff_at": cutoff.isoformat(),
            "event_metadata_source_ids": ["schedule"],
            "known_context": [],
        }

        def source(source_id, kind, text, publisher, **extra):
            published = now - timedelta(minutes=30)
            return {
                "schema_version": 1,
                "record_type": "idea-layer-source",
                "source_id": source_id,
                "canonical_source_id": f"canonical:{source_id}",
                "source_type": kind,
                "source_url": f"https://synthetic.invalid/{source_id}",
                "publisher": publisher,
                "published_at": published.isoformat(),
                "observed_at": now.isoformat(),
                "availability_status": "verified_before_cutoff",
                "availability_evidence": "synthetic fixture response hash",
                "timestamp_basis": "synthetic fixture timestamp",
                "content": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "reuse_terms": "Synthetic fixture only",
                **extra,
            }

        sources = [
            source("schedule", "event_metadata", "Synthetic event schedule: trade deal discussion", "Synthetic calendar"),
            source("news-one", "news", "Ukraine war peace deal talks continue", "News One"),
            source("news-two", "news", "Ukraine war peace deal remains uncertain", "News Two"),
            source("post-one", "trump_post", "Ukraine war must end with a peace deal", "Trump", account="synthetic-account", post_id="post-1"),
            source("news-other", "news", "Prime minister visits regional leaders", "News Three"),
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

    def test_proposals_are_lexical_cited_and_snapshot_bound(self):
        snapshot, now = self._snapshot()
        proposal_set = propose_headline_issue_phrases(snapshot, generated_at=now)
        info = validate_idea_proposal_set(proposal_set, snapshot)
        self.assertGreater(info["proposal_count"], 0)
        self.assertTrue(proposal_set["generator"].startswith(GENERATOR_VERSION + ";"))
        proposals = {row["idea_id"]: row for row in proposal_set["proposals"]}
        self.assertIn("ukraine war", proposals)
        self.assertIn("deal peace", proposals)
        for proposal in proposal_set["proposals"]:
            self.assertNotIn("schedule", proposal["evidence_source_ids"])
            self.assertTrue(set(proposal["evidence_source_ids"]) <= {"news-one", "news-two", "post-one", "news-other"})
        self.assertEqual(set(proposals["ukraine war"]["evidence_source_ids"]), {"news-one", "news-two", "post-one"})
        forecast = rank_pre_speech_ideas(snapshot, proposal_set, generated_at=now + timedelta(minutes=1))
        self.assertEqual(forecast["candidates"][0]["idea_id"], "deal peace")
        self.assertEqual(forecast["input_snapshot_sha256"], snapshot["snapshot_sha256"])

    def test_rejects_generation_at_or_after_cutoff(self):
        snapshot, _ = self._snapshot()
        cutoff = datetime.fromisoformat(snapshot["prediction_cutoff_at"])
        with self.assertRaisesRegex(ValueError, "strictly before prediction cutoff"):
            propose_headline_issue_phrases(snapshot, generated_at=cutoff)

    def test_unknown_terms_are_not_open_vocabulary_ideas(self):
        snapshot, now = self._snapshot()
        proposal_set = propose_headline_issue_phrases(snapshot, generated_at=now)
        surfaces = {row["idea"].casefold() for row in proposal_set["proposals"]}
        self.assertNotIn("prime minister", surfaces)
        self.assertIn("ukraine war", surfaces)

    def test_cli_writes_an_immutable_provider_neutral_proposal_set(self):
        snapshot, _ = self._snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            output_path = root / "proposals.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main([
                    "idea-headline-propose", "--snapshot", str(snapshot_path),
                    "--output", str(output_path),
                ]), 0)
            proposal_set = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(proposal_set["generator"].startswith(GENERATOR_VERSION + ";"))
            self.assertEqual(len(proposal_set["proposals"]), 3)
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main([
                    "idea-headline-propose", "--snapshot", str(snapshot_path),
                    "--output", str(output_path),
                ]), 2)


if __name__ == "__main__":
    unittest.main()
