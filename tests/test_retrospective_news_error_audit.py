import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speechprefix.headline_idea_matching import extract_headline_ideas
from speechprefix.idea_layer.retrospective_news_error_audit import audit_retrospective_news_forecast


def seal(report):
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    report["report_content_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


class RetrospectiveNewsErrorAuditTests(unittest.TestCase):
    def test_separates_lexical_stage_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshots.jsonl"
            forecast_path = root / "forecast.json"
            agreement_path = root / "agreement.json"
            reference_path = root / "reference.jsonl"
            output_path = root / "audit.json"
            sources = []
            for index in range(4):
                sources.append({
                    "title": "Tariffs rise amid trade talks",
                    "publisher": f"Publisher {index}",
                    "published_at": f"2025-04-{10 + index:02d}",
                    "source_url": f"https://example.test/tariffs/{index}",
                })
            sources.extend([
                {"title": "Ukraine war talks continue", "publisher": "U1", "published_at": "2025-04-20", "source_url": "https://example.test/ukraine"},
                {"title": "Navy recruiting morale improves", "publisher": "N1", "published_at": "2025-04-21", "source_url": "https://example.test/navy"},
            ])
            snapshot = {
                "event_id": "event-1", "event_date": "2025-05-01",
                "candidate_sources": sources,
            }
            snapshot_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
            inventory = extract_headline_ideas([
                {"title": x["title"], "publisher": x["publisher"], "published_at": x["published_at"], "source_url": x["source_url"]}
                for x in sources
            ], "2025-05-01")
            top = inventory[0]
            forecast = seal({
                "schema_version": 1,
                "artifact_type": "retrospective-news-issue-phrase-forecast-replay-reviewed-reference-subset",
                "input_snapshots_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                "forecasts": [{
                    "event_id": "event-1", "event_date": "2025-05-01",
                    "predictions": [{"idea_id": top["idea_id"], "idea": top["candidate_idea"], "rank": 1}],
                }],
            })
            forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
            forecast_file_sha = hashlib.sha256(forecast_path.read_bytes()).hexdigest()
            ideas = [
                ("hurricane_helene_disaster_response", "Provide disaster response after Hurricane Helene."),
                ("navy_recruiting_and_military_morale", "Highlight Navy recruiting and rising military morale."),
                ("ukraine_war_ceasefire", "Seek an end to the Ukraine war and a ceasefire."),
                ("tariff_revenue_deficit", "Use tariff revenue to reduce the federal deficit."),
            ]
            reference = {
                "event_id": "event-1", "event_date": "2025-05-01",
                "review_status": "assistant-reviewed-transcript-only",
                "outcome_ideas": [
                    {"idea_id": idea_id, "idea": text, "forecastable": True, "evidence_turn_ids": [f"turn-{i}"]}
                    for i, (idea_id, text) in enumerate(ideas, 1)
                ],
            }
            reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
            agreement = seal({
                "schema_version": 1,
                "forecast_report_sha256": forecast_file_sha,
                "events": [{
                    "event_id": "event-1", "event_date": "2025-05-01",
                    "outcome_idea_ids_without_match_or_partial": [idea_id for idea_id, _ in ideas],
                    "candidate_decisions": [{"candidate_id": top["idea_id"], "decision": "not_supported"}],
                }],
            })
            agreement_path.write_text(json.dumps(agreement), encoding="utf-8")
            report = audit_retrospective_news_forecast(
                snapshots_path=snapshot_path,
                forecast_path=forecast_path,
                agreement_path=agreement_path,
                reference_paths=[reference_path],
                output_path=output_path,
                top_k=1,
            )
            buckets = report["misses_by_lexical_diagnostic_bucket"]
            self.assertEqual(report["forecastable_outcome_miss_count"], 4)
            self.assertEqual(buckets["no_detectable_headline_term"], 1)
            self.assertEqual(buckets["headline_term_but_no_phrase_candidate_overlap"], 1)
            self.assertEqual(buckets["overlapping_phrase_candidate_below_top_k"], 1)
            self.assertEqual(buckets["top_k_lexical_overlap_but_not_semantically_matched"], 1)
            self.assertTrue(output_path.exists())
            with self.assertRaises(FileExistsError):
                audit_retrospective_news_forecast(
                    snapshots_path=snapshot_path, forecast_path=forecast_path,
                    agreement_path=agreement_path, reference_paths=[reference_path],
                    output_path=output_path,
                )


if __name__ == "__main__":
    unittest.main()
