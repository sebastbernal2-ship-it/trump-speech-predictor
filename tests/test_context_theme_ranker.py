import unittest

from speechprefix.context_theme_ranker import BernoulliThemeRanker, tokenize_headlines


class ContextThemeRankerTests(unittest.TestCase):
    def test_bernoulli_ranker_orders_context_cues_and_is_repeatable(self):
        docs = [
            {"ballot", "campaign"}, {"ballot", "candidate"},
            {"prices", "jobs"}, {"prices", "industry"},
        ]
        labels = [{"election"}, {"election"}, {"economy"}, {"economy"}]
        model = BernoulliThemeRanker().fit(docs, labels, ["election", "economy"])
        first = model.predict({"ballot", "campaign"})
        second = model.predict({"ballot", "campaign"})
        self.assertEqual(first, second)
        self.assertEqual(first[0]["theme_family"], "election")
        self.assertEqual(first[0]["score_type"], "uncalibrated_bernoulli_nb_posterior")

    def test_headline_features_exclude_event_title(self):
        tokens = tokenize_headlines({
            "event_title_metadata_only": "LEAKED EVENT TITLE",
            "candidate_sources": [{"title": "Nevada voters and early turnout"}],
        })
        self.assertIn("nevada", tokens)
        self.assertNotIn("leaked", tokens)
        self.assertNotIn("event", tokens)

    def test_ranker_rejects_unknown_labels(self):
        model = BernoulliThemeRanker()
        with self.assertRaisesRegex(ValueError, "unknown theme family"):
            model.fit([{"jobs"}], [{"economy", "unknown"}], ["economy"])


if __name__ == "__main__":
    unittest.main()
