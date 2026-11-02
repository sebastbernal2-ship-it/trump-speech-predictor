import unittest

from speechprefix.hybrid import Candidate, combine_candidate_mass


class HybridFusionTests(unittest.TestCase):
    def test_combines_candidate_scores_and_marginalizes_next_token(self):
        candidates = [
            Candidate("win now", {"llm": -0.2, "ncm": -0.3}),
            Candidate("win again", {"llm": -0.4, "ncm": -0.5}),
            Candidate("work together", {"llm": -1.2, "ncm": -0.8}),
        ]
        distribution = dict(combine_candidate_mass(candidates, {"llm": 0.6, "ncm": 0.4}))
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution["win"], distribution["work"])

    def test_distinguishes_punctuation_first_tokens(self):
        candidates = [Candidate(", then", {"llm": 0.0}), Candidate("however", {"llm": -1.0})]
        distribution = dict(combine_candidate_mass(candidates, {"llm": 1.0}))
        self.assertEqual(set(distribution), {",", "however"})

    def test_rejects_missing_component_score(self):
        with self.assertRaisesRegex(ValueError, "missing component scores"):
            combine_candidate_mass([Candidate("hello", {"llm": -0.1})], {"llm": 1, "ncm": 1})

    def test_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            combine_candidate_mass([Candidate("hello", {"llm": 0.0})], {"llm": 1.0}, temperature=0)


if __name__ == "__main__":
    unittest.main()
