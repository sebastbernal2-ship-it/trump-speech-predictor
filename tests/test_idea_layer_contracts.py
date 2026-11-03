import hashlib
import unittest

from speechprefix.idea_layer.contracts import (
    is_source_eligible_at,
    seal_pre_speech_forecast,
    validate_event_manifest,
    validate_idea_update,
    validate_pre_speech_forecast,
    validate_source_evidence,
)


class IdeaLayerContractsTests(unittest.TestCase):
    def _source(self, source_id, source_type, content, *, published, observed, **extra):
        row = {
            "schema_version": 1,
            "record_type": "idea-layer-source",
            "source_id": source_id,
            "source_type": source_type,
            "source_url": f"https://example.test/{source_id}",
            "publisher": "Example publisher",
            "published_at": published,
            "observed_at": observed,
            "availability_status": "verified_before_cutoff",
            "availability_evidence": "fixture evidence sha256:" + "0" * 64,
            "timestamp_basis": "fixture publisher timestamp field",
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "reuse_terms": "Local research use; retain source attribution.",
        }
        row.update(extra)
        return row

    def _manifest(self):
        return {
            "schema_version": 1,
            "record_type": "idea-layer-event",
            "event_id": "speech-2030-01-20",
            "event_start_at": "2030-01-20T18:00:00Z",
            "prediction_cutoff_at": "2030-01-20T16:00:00Z",
            "event_metadata_source_ids": ["schedule"],
            "known_context": [{
                "fact_id": "fact-oil",
                "fact": "A major oil disruption is in the news.",
                "source_ids": ["news-oil"],
            }],
        }

    def _sources(self):
        return [
            self._source(
                "schedule", "event_metadata", "Official event schedule",
                published="2030-01-19T10:00:00Z", observed="2030-01-19T10:05:00Z",
            ),
            self._source(
                "news-oil", "news", "Oil supply disruption headline",
                published="2030-01-20T12:00:00Z", observed="2030-01-20T12:10:00Z",
            ),
            self._source(
                "trump-post", "trump_post", "We are watching the situation.",
                published="2030-01-20T13:00:00Z", observed="2030-01-20T13:02:00Z",
                account="realDonaldTrump", post_id="post-123",
            ),
        ]

    def _forecast(self):
        return seal_pre_speech_forecast({
            "schema_version": 1,
            "record_type": "idea-layer-forecast",
            "forecast_id": "forecast-2030-01-20-v0",
            "input_snapshot_sha256": "a" * 64,
            "proposal_set_id": "proposal-set-2030-01-20",
            "proposal_set_sha256": "b" * 64,
            "proposal_generator": "test-fixture",
            "event_id": "speech-2030-01-20",
            "prediction_cutoff_at": "2030-01-20T16:00:00Z",
            "input_snapshot_created_at": "2030-01-20T15:50:00Z",
            "generated_at": "2030-01-20T15:55:00Z",
            "model_version": "idea-context-v0",
            "input_source_ids": ["schedule", "news-oil", "trump-post"],
            "candidates": [{
                "rank": 1,
                "idea_id": "issue:oil-supply",
                "idea": "Oil supply and prices",
                "score": 0.72,
                "score_kind": "ranking_score",
                "evidence_source_ids": ["news-oil", "trump-post"],
            }],
        })

    def test_verified_source_requires_explicit_timestamp_and_availability_evidence(self):
        source = self._sources()[1]
        source.pop("availability_evidence")
        with self.assertRaisesRegex(ValueError, "availability_evidence"):
            validate_source_evidence(source)
        source["availability_status"] = "unknown"
        self.assertEqual(validate_source_evidence(source)["source_id"], "news-oil")

    def test_source_fingerprint_and_as_of_availability(self):
        source = self._sources()[1]
        validate_source_evidence(source)
        self.assertTrue(is_source_eligible_at(source, "2030-01-20T16:00:00Z"))
        self.assertFalse(is_source_eligible_at(source, "2030-01-20T12:05:00Z"))
        source["content"] += " edited"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_source_evidence(source)

    def test_post_requires_platform_identity_and_verified_timestamped_source(self):
        post = self._sources()[2]
        self.assertTrue(is_source_eligible_at(post, "2030-01-20T16:00:00Z"))
        del post["post_id"]
        with self.assertRaisesRegex(ValueError, "post_id"):
            validate_source_evidence(post)

    def test_pre_speech_forecast_is_grounded_in_only_verified_pre_cutoff_sources(self):
        manifest = self._manifest()
        self.assertEqual(validate_event_manifest(manifest)["event_id"], manifest["event_id"])
        result = validate_pre_speech_forecast(self._forecast(), manifest, self._sources())
        self.assertTrue(result["strict_pre_cutoff"])
        late_source = self._source(
            "late", "news", "Published too late", published="2030-01-20T16:00:00Z",
            observed="2030-01-20T16:01:00Z",
        )
        forecast = self._forecast()
        forecast["input_source_ids"].append("late")
        with self.assertRaisesRegex(ValueError, "unavailable or unverified"):
            validate_pre_speech_forecast(forecast, manifest, self._sources() + [late_source])

    def test_event_metadata_citations_must_point_to_event_metadata_records(self):
        manifest = self._manifest()
        manifest["event_metadata_source_ids"] = ["news-oil"]
        with self.assertRaisesRegex(ValueError, "must reference event_metadata"):
            validate_pre_speech_forecast(self._forecast(), manifest, self._sources())

    def test_manifest_and_forecast_reject_bad_cutoffs_and_target_leakage(self):
        manifest = self._manifest()
        manifest["prediction_cutoff_at"] = manifest["event_start_at"]
        with self.assertRaisesRegex(ValueError, "strictly before"):
            validate_event_manifest(manifest)
        forecast = self._forecast()
        forecast["debug"] = {"target_ideas": ["oil"]}
        with self.assertRaisesRegex(ValueError, "forbidden outcome"):
            validate_pre_speech_forecast(forecast, self._manifest(), self._sources())

    def test_probability_requires_calibration_provenance(self):
        forecast = self._forecast()
        forecast["candidates"][0]["score_kind"] = "probability"
        with self.assertRaisesRegex(ValueError, "calibration_ref"):
            validate_pre_speech_forecast(forecast, self._manifest(), self._sources())
        forecast["candidates"][0]["calibration_ref"] = "calibration-report-sha256"
        forecast = seal_pre_speech_forecast(forecast)
        self.assertTrue(validate_pre_speech_forecast(forecast, self._manifest(), self._sources())["strict_pre_cutoff"])

    def test_live_update_can_only_raise_ideas_from_already_visible_turns(self):
        update = {
            "schema_version": 1,
            "record_type": "idea-layer-update",
            "event_id": "speech-live",
            "forecast_id": "forecast-001",
            "updated_at": "2030-01-20T18:05:00Z",
            "observed_turn_ids": ["turn-001"],
            "changes": [{
                "idea_id": "issue:gaza",
                "action": "raised",
                "evidence_turn_ids": ["turn-001"],
            }],
        }
        self.assertEqual(validate_idea_update(update, ["turn-001"])["change_count"], 1)
        update["changes"][0]["evidence_turn_ids"] = ["turn-002"]
        with self.assertRaisesRegex(ValueError, "unobserved turn"):
            validate_idea_update(update, ["turn-001"])


if __name__ == "__main__":
    unittest.main()
