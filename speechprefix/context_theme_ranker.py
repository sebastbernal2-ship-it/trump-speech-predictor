"""Weakly supervised Bernoulli naive-Bayes ranker for broad theme families.

The model consumes dated pre-event headlines. Current targets are assistant-
authored drafts; scores are diagnostic and uncalibrated, not semantic probabilities.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

TOKEN = re.compile(r"[a-z][a-z']{1,}")
STOPWORDS = frozenset("""a an and are as at be been being by for from had has have he her his i in into is it its of on or our she that the their them there they this to was were will with we you your""".split())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read local JSONL objects and report malformed rows with their line number."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from None
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def tokenize_headlines(snapshot: dict[str, Any]) -> set[str]:
    """Build features from source headlines only, never event title or transcript."""
    sources = snapshot.get("candidate_sources")
    if not isinstance(sources, list):
        raise ValueError(f"invalid candidate_sources for {snapshot.get('event_id')}")
    tokens: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"invalid context source for {snapshot.get('event_id')}")
        title = source.get("title")
        if not isinstance(title, str):
            raise ValueError(f"context source has no title for {snapshot.get('event_id')}")
        tokens.update(word for word in TOKEN.findall(title.casefold()) if word not in STOPWORDS)
    return tokens


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700)))
    exp_value = math.exp(max(value, -700))
    return exp_value / (1.0 + exp_value)


class BernoulliThemeRanker:
    """Independent Bernoulli naive-Bayes classifiers, one per broad theme family."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.categories: list[str] = []
        self.vocabulary: list[str] = []
        self.category_counts: dict[str, int] = {}
        self.log_prior_odds: dict[str, float] = {}
        self.log_present_ratio: dict[str, dict[str, float]] = {}
        self.log_absent_ratio: dict[str, dict[str, float]] = {}

    def fit(self, documents: list[set[str]], labels: list[set[str]], categories: list[str]):
        if not documents or len(documents) != len(labels):
            raise ValueError("training documents and labels must have equal nonzero length")
        if not categories or len(set(categories)) != len(categories):
            raise ValueError("categories must be a nonempty list of unique names")
        if any(not target.issubset(categories) for target in labels):
            raise ValueError("training labels contain an unknown theme family")
        self.categories = list(categories)
        self.vocabulary = sorted(set().union(*documents))
        if not self.vocabulary:
            raise ValueError("training context has no headline tokens")
        n = len(documents)
        self.category_counts = {}
        self.log_prior_odds = {}
        self.log_present_ratio = {}
        self.log_absent_ratio = {}
        for category in self.categories:
            positive_indices = [index for index, target in enumerate(labels) if category in target]
            positive_count = len(positive_indices)
            negative_count = n - positive_count
            self.category_counts[category] = positive_count
            self.log_prior_odds[category] = math.log((positive_count + self.alpha) / (negative_count + self.alpha))
            present_counts = Counter(token for index in positive_indices for token in documents[index])
            negative_present_counts = Counter(
                token for index, target in enumerate(labels) if category not in target for token in documents[index]
            )
            pos_denominator = positive_count + 2 * self.alpha
            neg_denominator = negative_count + 2 * self.alpha
            present_ratio = {}
            absent_ratio = {}
            for token in self.vocabulary:
                p_pos = (present_counts[token] + self.alpha) / pos_denominator
                p_neg = (negative_present_counts[token] + self.alpha) / neg_denominator
                present_ratio[token] = math.log(p_pos / p_neg)
                absent_ratio[token] = math.log((1 - p_pos) / (1 - p_neg))
            self.log_present_ratio[category] = present_ratio
            self.log_absent_ratio[category] = absent_ratio
        return self

    def predict(self, document: set[str]) -> list[dict[str, Any]]:
        if not self.categories:
            raise ValueError("fit the ranker before predicting")
        known = document.intersection(self.vocabulary)
        ranked = []
        for category in self.categories:
            score = self.log_prior_odds[category]
            contributions = []
            for token in self.vocabulary:
                contribution = (
                    self.log_present_ratio[category][token]
                    if token in known else self.log_absent_ratio[category][token]
                )
                score += contribution
                if token in known and contribution > 0:
                    contributions.append((contribution, token))
            ranked.append({
                "theme_family": category,
                "score": _sigmoid(score),
                "score_type": "uncalibrated_bernoulli_nb_posterior",
                "positive_feature_tokens": [token for _, token in sorted(contributions, reverse=True)[:8]],
            })
        return sorted(ranked, key=lambda item: (-item["score"], item["theme_family"]))
