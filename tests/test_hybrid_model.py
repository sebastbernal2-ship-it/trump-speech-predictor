import unittest

from speechprefix.hybrid import Candidate
from speechprefix.hybrid_model import LLMNCMHybrid


class FakeLLM:
    def generate_candidates(self, prefix, count):
        return [Candidate("we agree", {"llm": -0.2})]

    def score_candidates(self, prefix, candidates):
        return [-0.2 if text == "we agree" else -0.5 for text in candidates]


class FakeNCM:
    def generate(self, count):
        return ["great people", "we agree"]

    def score_candidates(self, candidates):
        return [-0.4 for _ in candidates]


class HybridModelTests(unittest.TestCase):
    def test_unions_proposals_scores_every_candidate_and_returns_mass(self):
        hybrid = LLMNCMHybrid(FakeLLM(), FakeNCM())
        candidates, mass = hybrid.predict("prefix", llm_count=1, ncm_count=2)
        self.assertEqual([candidate.text for candidate in candidates], ["we agree", "great people"])
        self.assertEqual(set(candidates[0].scores), {"llm", "ncm"})
        self.assertAlmostEqual(sum(value for _, value in mass), 1.0)

    def test_rejects_empty_proposal_pool(self):
        class EmptyLLM(FakeLLM):
            def generate_candidates(self, prefix, count):
                return []

        class EmptyNCM(FakeNCM):
            def generate(self, count):
                return []

        with self.assertRaisesRegex(RuntimeError, "empty candidate pool"):
            LLMNCMHybrid(EmptyLLM(), EmptyNCM()).predict("prefix")


if __name__ == "__main__":
    unittest.main()
