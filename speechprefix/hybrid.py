"""Small candidate-level fusion primitive for future LLM/NCM adapters.

The returned values are normalized over a finite candidate pool only. They are
not calibrated next-word probabilities and must not be labeled confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .model import tokenize


@dataclass(frozen=True)
class Candidate:
    """A decoded continuation and normalized component scores for that same text."""

    text: str
    scores: Mapping[str, float]


def combine_candidate_mass(
    candidates: list[Candidate],
    weights: Mapping[str, float],
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """Combine candidate scores, then marginalize finite candidate mass by first token.

    Each backend must score every candidate using its own documented score
    normalization. Fusion weights and temperature must be selected on validation
    data, not tuned on the test speeches.
    """
    if not candidates:
        raise ValueError("candidate pool must not be empty")
    if not weights or any(not math.isfinite(weight) for weight in weights.values()):
        raise ValueError("finite model weights are required")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")

    logits: list[tuple[str, float]] = []
    for candidate in candidates:
        tokens = tokenize(candidate.text)
        if not tokens:
            raise ValueError("candidate continuation must contain a token")
        missing = set(weights) - candidate.scores.keys()
        if missing:
            raise ValueError(f"candidate is missing component scores: {', '.join(sorted(missing))}")
        if any(not math.isfinite(candidate.scores[name]) for name in weights):
            raise ValueError("component scores must be finite")
        logits.append((tokens[0], sum(weights[name] * candidate.scores[name] for name in weights) / temperature))

    maximum = max(score for _, score in logits)
    masses = [(token, math.exp(score - maximum)) for token, score in logits]
    total = sum(mass for _, mass in masses)
    by_token: dict[str, float] = {}
    for token, mass in masses:
        by_token[token] = by_token.get(token, 0.0) + mass / total
    return sorted(by_token.items(), key=lambda item: (-item[1], item[0]))
