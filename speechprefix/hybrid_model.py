"""Experimental NCM proposal + full-prefix LLM reranking pipeline."""

from __future__ import annotations

from .hybrid import Candidate, combine_candidate_mass
from .llm import CausalLM
from .ncm import NCMGenerator


class LLMNCMHybrid:
    """Combine unconditional NCM proposals with context-aware LLM scores.

    Component scores are mean log-likelihoods over different units (NCM codes
    vs. GPT-2 tokens), so weights are provisional and must be fitted on a
    speech-level validation set before interpreting the resulting mass.
    """

    def __init__(self, llm: CausalLM | None = None, ncm: NCMGenerator | None = None):
        self.llm = llm or CausalLM()
        self.ncm = ncm or NCMGenerator()

    def predict(
        self,
        prefix: str,
        llm_count: int = 4,
        ncm_count: int = 4,
        weights: dict[str, float] | None = None,
    ) -> tuple[list[Candidate], list[tuple[str, float]]]:
        """Return the finite scored candidate pool and first-token candidate mass."""
        llm_proposals = self.llm.generate_candidates(prefix, count=llm_count)
        ncm_proposals = self.ncm.generate(count=ncm_count)
        texts = list(dict.fromkeys(
            [item.text.strip() for item in llm_proposals]
            + [item.strip() for item in ncm_proposals]
        ))
        if not texts:
            raise RuntimeError("both backends returned an empty candidate pool")

        llm_scores = self.llm.score_candidates(prefix, texts)
        ncm_scores = self.ncm.score_candidates(texts)
        candidates = [
            Candidate(text, {"llm": llm_score, "ncm": ncm_score})
            for text, llm_score, ncm_score in zip(texts, llm_scores, ncm_scores)
        ]
        selected_weights = weights or {"llm": 0.5, "ncm": 0.5}
        return candidates, combine_candidate_mass(candidates, selected_weights)
