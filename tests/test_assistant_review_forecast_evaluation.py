import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speechprefix.idea_layer.assistant_review_forecast_evaluation import evaluate_assistant_reviewed_forecasts


class AssistantReviewForecastEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.forecast_path = self.root / "forecast.json"
        self.reference_path = self.root / "reference.jsonl"
        self.decisions_path = self.root / "decisions.jsonl"
        self.protocol_path = self.root / "policy.md"
        self._write_fixtures()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_jsonl(path, rows):
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _write_fixtures(self):
        self.protocol_path.write_text("Test review protocol.\n", encoding="utf-8")
        forecast = {
            "schema_version": 1,
            "artifact_type": "assistant-reference-review-history-only-forecast-replay",
            "forecasts": [{
                "event_id": "event-1",
                "event_date": "2030-01-01",
                "predictions": [
                    {"rank": 1, "idea_id": "topic one", "idea": "Topic one"},
                    {"rank": 2, "idea_id": "topic two", "idea": "Topic two"},
                ],
            }],
        }
        forecast["report_content_sha256"] = hashlib.sha256(
            json.dumps(forecast, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
        self.reference = {
            "event_id": "event-1",
            "event_date": "2030-01-01",
            "review_status": "assistant-reviewed-transcript-only",
            "source_record_sha256": "c" * 64,
            "target_transcript_sha256": "b" * 64,
            "coverage_assessment": "full available transcript read",
            "outcome_ideas": [
                {"idea_id": "idea-one", "forecastable": True, "evidence_turn_ids": ["turn-1"]},
                {"idea_id": "idea-two", "forecastable": True, "evidence_turn_ids": ["turn-2"]},
            ],
        }
        self._write_jsonl(self.reference_path, [self.reference])
        self._write_decisions()

    def _write_decisions(self, *, bad_citation=False, omit_candidate=False):
        forecast_sha = hashlib.sha256(self.forecast_path.read_bytes()).hexdigest()
        decisions = [
            {
                "candidate_id": "topic one",
                "rank": 1,
                "candidate_idea": "Topic one",
                "decision": "match",
                "matched_outcome_idea_ids": ["idea-one"],
                "evidence_turn_ids": ["wrong-turn" if bad_citation else "turn-1"],
                "rationale": "The transcript directly supports this topic.",
            },
            {
                "candidate_id": "topic two",
                "rank": 2,
                "candidate_idea": "Topic two",
                "decision": "not_supported",
                "matched_outcome_idea_ids": [],
                "evidence_turn_ids": ["turn-2"],
                "rationale": "The cited transcript turn addresses a different idea; no forecastable outcome supports this candidate."
            },
        ]
        row = {
            "event_id": "event-1",
            "event_date": "2030-01-01",
            "review_status": "assistant-reviewed-posthoc-nonblind",
            "forecast_report_sha256": forecast_sha,
            "source_record_sha256": "c" * 64,
            "target_transcript_sha256": "b" * 64,
            "reviewer": {
                "reviewer_type": "AI assistant",
                "protocol_sha256": hashlib.sha256(self.protocol_path.read_bytes()).hexdigest(),
            },
            "decisions": decisions[:1] if omit_candidate else decisions,
        }
        self._write_jsonl(self.decisions_path, [row])

    def test_scores_strict_matches_and_misses_with_provenance(self):
        report = evaluate_assistant_reviewed_forecasts(
            forecast_path=self.forecast_path,
            reference_paths=[self.reference_path],
            decisions_path=self.decisions_path,
            protocol_snapshot_path=self.protocol_path,
        )
        self.assertEqual(report["scored_event_count"], 1)
        self.assertEqual(report["candidate_decision_counts"]["match"], 1)
        self.assertEqual(report["candidate_decision_counts"]["not_supported"], 1)
        self.assertEqual(report["outcome_coverage"]["strict_matched_count"], 1)
        self.assertEqual(report["outcome_coverage"]["unmatched_count"], 1)
        self.assertEqual(report["events"][0]["outcome_idea_ids_without_match_or_partial"], ["idea-two"])

    def test_rejects_missing_candidate_decision(self):
        self._write_decisions(omit_candidate=True)
        with self.assertRaisesRegex(ValueError, "decisions do not cover exactly"):
            evaluate_assistant_reviewed_forecasts(
                forecast_path=self.forecast_path,
                reference_paths=[self.reference_path],
                decisions_path=self.decisions_path,
                protocol_snapshot_path=self.protocol_path,
            )

    def test_rejects_citation_not_in_matched_outcome(self):
        self._write_decisions(bad_citation=True)
        with self.assertRaisesRegex(ValueError, "citations are not present"):
            evaluate_assistant_reviewed_forecasts(
                forecast_path=self.forecast_path,
                reference_paths=[self.reference_path],
                decisions_path=self.decisions_path,
                protocol_snapshot_path=self.protocol_path,
            )

    def test_news_forecast_uses_news_specific_evaluation_limitations(self):
        forecast = json.loads(self.forecast_path.read_text())
        forecast["artifact_type"] = "retrospective-news-issue-phrase-forecast-replay"
        forecast["model_family"] = "headline phrase baseline"
        forecast.pop("report_content_sha256")
        forecast["report_content_sha256"] = hashlib.sha256(
            json.dumps(forecast, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
        self._write_decisions()
        report = evaluate_assistant_reviewed_forecasts(
            forecast_path=self.forecast_path,
            reference_paths=[self.reference_path],
            decisions_path=self.decisions_path,
            protocol_snapshot_path=self.protocol_path,
        )
        self.assertIn("news-to-idea", report["evaluation_mode"])
        self.assertIn("retrospective search-index coverage", report["limitations"][2])
        self.assertNotIn("fixed history baseline", report["limitations"][2])

    def test_semantic_bundle_gets_semantic_evaluation_label_and_limitations(self):
        forecast = json.loads(self.forecast_path.read_text())
        forecast["artifact_type"] = "semantic-idea-forecast-evaluation-bundle"
        forecast["provider"] = "synthetic-provider"
        forecast["model"] = "synthetic-model-v1"
        forecast["ranker_version"] = "explicit-evidence-signals-v1"
        forecast.pop("report_content_sha256", None)
        forecast["report_content_sha256"] = hashlib.sha256(
            json.dumps(forecast, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
        self._write_decisions()
        report = evaluate_assistant_reviewed_forecasts(
            forecast_path=self.forecast_path,
            reference_paths=[self.reference_path],
            decisions_path=self.decisions_path,
            protocol_snapshot_path=self.protocol_path,
        )
        self.assertIn("semantic event-conditioned", report["evaluation_mode"])
        self.assertIn("uncalibrated", report["limitations"][2])

    def test_rejects_tampered_forecast_report(self):
        forecast = json.loads(self.forecast_path.read_text())
        forecast["forecasts"][0]["predictions"][0]["idea"] = "Tampered"
        self.forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hash"):
            evaluate_assistant_reviewed_forecasts(
                forecast_path=self.forecast_path,
                reference_paths=[self.reference_path],
                decisions_path=self.decisions_path,
                protocol_snapshot_path=self.protocol_path,
            )


if __name__ == "__main__":
    unittest.main()
