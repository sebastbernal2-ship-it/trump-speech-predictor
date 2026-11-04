import unittest
from collections import Counter
from datetime import date, timedelta

from speechprefix.idea_layer.news_phrase_backtest import (
    _fit_bernoulli_nb,
    _fit_multinomial_nb,
    _graph_label_propagation_rank,
    _headline_features,
    _hybrid_lexical_rank,
    _query_conditioned_type_path_rank,
    _tfidf_knn_rank,
    _score,
    _score_multinomial,
    evaluate_retrospective_news_phrase_model,
)


class NewsPhraseBacktestTests(unittest.TestCase):
    def test_bernoulli_model_learns_headline_to_phrase_association(self):
        train = [
            {"features": {"u:tariff", "u:trade"}, "target_ids": {"trade deficit"}},
            {"features": {"u:tariff", "u:trade", "b:trade_deal"}, "target_ids": {"trade deficit"}},
            {"features": {"u:health", "u:care"}, "target_ids": {"health care"}},
            {"features": {"u:health", "u:hospital"}, "target_ids": {"health care"}},
        ]
        _, model = _fit_bernoulli_nb(train)
        scores = _score(model, {"u:tariff", "u:trade"})
        self.assertGreater(scores["trade deficit"], scores["health care"])

    def test_bernoulli_min_document_frequency_drops_single_event_terms(self):
        train = [
            {"features": {"u:common", "u:rare"}, "target_ids": {"trade deficit"}},
            {"features": {"u:common"}, "target_ids": {"health care"}},
        ]
        vocabulary, _ = _fit_bernoulli_nb(train, min_document_frequency=2)
        self.assertEqual(vocabulary, ["u:common"])

    def test_multinomial_model_uses_positive_headline_word_counts(self):
        train = [
            {"feature_counts": {"u:tariff": 2, "u:trade": 1}, "target_ids": {"trade deficit"}},
            {"feature_counts": {"u:tariff": 1, "u:trade": 1}, "target_ids": {"trade deficit"}},
            {"feature_counts": {"u:hospital": 2, "u:health": 1}, "target_ids": {"health care"}},
            {"feature_counts": {"u:hospital": 1, "u:health": 1}, "target_ids": {"health care"}},
        ]
        model = _fit_multinomial_nb(train)
        scores = _score_multinomial(model, Counter({"u:tariff": 2, "u:trade": 1}))
        self.assertGreater(scores["trade deficit"], scores["health care"])

    def test_hybrid_requires_phrase_terms_and_cites_matching_headlines(self):
        train = [
            {"target_ids": {"cut tax"}, "label_surfaces": {"cut tax": "Tax cuts"}},
            {"target_ids": {"cut tax"}, "label_surfaces": {"cut tax": "Tax cuts"}},
            {"target_ids": {"border security"}, "label_surfaces": {"border security": "Border security"}},
        ]
        sources = [{
            "title": "Congress debates tax cuts",
            "publisher": "Synthetic News",
            "published_at": "2020-01-01",
            "source_url": "https://synthetic.invalid/tax",
            "_features": {"u:congress", "u:debates", "u:tax", "u:cuts"},
        }]
        event = {"sources": sources}
        ranked = _hybrid_lexical_rank(event, train, {"cut tax", "border security"})
        self.assertEqual([row["idea_id"] for row in ranked], ["cut tax"])
        self.assertEqual(ranked[0]["idea"], "Tax cuts")
        self.assertEqual(ranked[0]["supporting_headlines"][0]["source_url"], "https://synthetic.invalid/tax")

    def test_tfidf_knn_transfers_ideas_from_similar_prior_events(self):
        from datetime import date

        train = [
            {"event_id": "tariff-1", "date": date(2020, 1, 1), "features": {"u:tariff"},
             "target_ids": {"trade deficit"}, "label_surfaces": {"trade deficit": "Trade deficit"}},
            {"event_id": "tariff-2", "date": date(2020, 2, 1), "features": {"u:tariff"},
             "target_ids": {"trade deficit"}, "label_surfaces": {"trade deficit": "Trade deficit"}},
            {"event_id": "health-1", "date": date(2020, 3, 1), "features": {"u:hospital"},
             "target_ids": {"health care"}, "label_surfaces": {"health care": "Health care"}},
        ]
        event = {"features": {"u:tariff"}, "sources": [{
            "title": "Tariff news", "_features": {"u:tariff"}, "published_at": "2021-01-01",
            "source_url": "https://synthetic.invalid/tariff",
        }]}
        ranked, neighbors = _tfidf_knn_rank(event, train, {"trade deficit", "health care"}, neighbor_count=2)
        self.assertEqual(ranked[0]["idea_id"], "trade deficit")
        self.assertEqual([row["event_id"] for row in neighbors], ["tariff-1", "tariff-2"])
        self.assertEqual(ranked[0]["supporting_headlines"][0]["source_url"], "https://synthetic.invalid/tariff")

    def test_heterogeneous_graph_propagates_labels_from_similar_contexts(self):
        train = [
            {"event_id": "tariff-1", "date": date(2020, 1, 1), "features": {"u:tariff"},
             "target_ids": {"trade deficit"}, "label_surfaces": {"trade deficit": "Trade deficit"}},
            {"event_id": "tariff-2", "date": date(2020, 2, 1), "features": {"u:tariff"},
             "target_ids": {"trade deficit"}, "label_surfaces": {"trade deficit": "Trade deficit"}},
            {"event_id": "health-1", "date": date(2020, 3, 1), "features": {"u:hospital"},
             "target_ids": {"health care"}, "label_surfaces": {"health care": "Health care"}},
        ]
        event = {"event_id": "query", "date": date(2021, 1, 1), "features": {"u:tariff"},
                 "target_ids": set(), "sources": [{"title": "Tariff news", "_features": {"u:tariff"},
                 "published_at": "2020-12-31", "source_url": "https://synthetic.invalid/tariff"}]}
        ranked, neighbors, mode = _graph_label_propagation_rank(event, train, {"trade deficit", "health care"})
        self.assertEqual(mode, "personalized_pagerank")
        self.assertEqual(ranked[0]["idea_id"], "trade deficit")
        self.assertEqual(neighbors[0]["event_id"], "tariff-1")
        self.assertEqual(ranked[0]["supporting_headlines"][0]["source_url"], "https://synthetic.invalid/tariff")

    def test_query_conditioning_changes_candidate_ranking_with_event_context(self):
        train = [
            {"target_ids": {"drug price"}, "label_surfaces": {"drug price": "Drug prices"}, "features": {"u:drug", "u:price"}},
            {"target_ids": {"border security"}, "label_surfaces": {"border security": "Border security"}, "features": {"u:border", "u:security"}},
        ]
        history = [
            {"idea_id": "drug price", "similar_event_vote": 0.5, "supporting_training_events": []},
            {"idea_id": "border security", "similar_event_vote": 0.5, "supporting_training_events": []},
        ]
        drug_event = {"sources": [{"title": "Drug prices", "published_at": "2024-01-01", "_features": {"u:drug", "u:prices"}}]}
        border_event = {"sources": [{"title": "Border security", "published_at": "2024-01-01", "_features": {"u:border", "u:security"}}]}
        drug_rank = _query_conditioned_type_path_rank(drug_event, train, {"drug price", "border security"}, history)
        border_rank = _query_conditioned_type_path_rank(border_event, train, {"drug price", "border security"}, history)
        self.assertEqual(drug_rank[0]["idea_id"], "drug price")
        self.assertEqual(border_rank[0]["idea_id"], "border security")
        self.assertGreater(drug_rank[0]["direct_query_path_score"], drug_rank[1]["direct_query_path_score"])

    def test_headline_features_exclude_same_day_and_later_sources(self):
        event_date = date(2026, 1, 10)
        features, eligible = _headline_features([
            {"title": "Trade Deficit and Tariff Policy", "published_at": "2026-01-09", "source_url": "https://news.test/old"},
            {"title": "Same Day Energy Prices", "published_at": "2026-01-10"},
            {"title": "Later Energy Prices", "published_at": "2026-01-11"},
        ], event_date)
        self.assertEqual(len(eligible), 1)
        self.assertIn("u:tariff", features)
        self.assertNotIn("u:energy", features)

    def test_retro_backtest_is_chronological_and_explicitly_limited(self):
        snapshots, events = [], []
        start = date(2020, 1, 1)
        for index in range(10):
            event_id = f"synthetic-{index:02d}"
            day = start + timedelta(days=index * 10)
            target_phrase = "drug prices" if index % 2 == 0 else "border security"
            headline_phrase = "drug prices" if index % 2 == 0 else "border security"
            snapshots.append({
                "event_id": event_id,
                "event_date": day.isoformat(),
                "retrieved_at": "2026-11-03T13:00:00+00:00",
                "candidate_sources": [{
                    "title": f"Trump discusses {headline_phrase} policy",
                    "publisher": "Synthetic News",
                    "published_at": (day - timedelta(days=1)).isoformat(),
                    "source_url": f"https://synthetic.invalid/{index}",
                }],
            })
            events.append({
                "event_id": event_id,
                "event_date": day.isoformat(),
                "target_speaker": "Donald Trump",
                "turns": [{
                    "turn_id": f"{event_id}:turn:1",
                    "speaker_role": "target_speaker",
                    "speaker_name": "Donald Trump",
                    "text": f"We are focused on {target_phrase}.",
                }],
            })

        class TemporaryInputs:
            def __init__(self, rows):
                import json
                import tempfile
                from pathlib import Path
                self._temp = tempfile.TemporaryDirectory()
                self.path = Path(self._temp.name) / "rows.jsonl"
                self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            def __enter__(self):
                return self.path
            def __exit__(self, *_):
                self._temp.cleanup()

        with TemporaryInputs(snapshots) as snapshot_path, TemporaryInputs(events) as event_path:
            report = evaluate_retrospective_news_phrase_model(snapshot_path, event_path)
        self.assertEqual(len(report["split"]["train_event_ids"]), 8)
        self.assertEqual(len(report["split"]["validation_event_ids"]), 1)
        self.assertEqual(len(report["split"]["test_event_ids"]), 1)
        self.assertTrue(report["evaluation_mode"].startswith("retrospective-only"))
        self.assertEqual(report["test_metrics"]["news_phrase_bernoulli_naive_bayes"]["events"], 1)
        self.assertTrue(report["limitations"])
        self.assertIn("aggregate_error_breakdown_by_model", report["test_diagnostics"])
        event_diagnostics = report["test_diagnostics"]["per_event"][0]
        self.assertIn("missed_weak_targets", event_diagnostics["models"]["bernoulli_nb"]["at_k"]["4"])
        self.assertIn("model_rankings", event_diagnostics)
        self.assertIn("headline_proposals_plus_history", event_diagnostics["models"])
        self.assertGreater(report["test_metrics"]["headline_proposals_plus_history"]["headline_proposal_target_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
