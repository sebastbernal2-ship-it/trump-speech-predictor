"""A small transparent word n-gram next-word predictor."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


@dataclass
class NGramModel:
    order: int
    counts: dict[tuple[str, ...], Counter[str]]
    vocabulary: set[str]

    @classmethod
    def train(cls, speeches: Iterable[str], order: int = 5) -> "NGramModel":
        if order < 1:
            raise ValueError("order must be at least 1")
        counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()
        for speech in speeches:
            words = tokenize(speech)
            vocabulary.update(words)
            padded = ["<s>"] * (order - 1) + words + ["</s>"]
            for i in range(order - 1, len(padded)):
                target = padded[i]
                for n in range(1, order + 1):
                    context = tuple(padded[i - n + 1 : i]) if n > 1 else ()
                    counts[context][target] += 1
        if not vocabulary:
            raise ValueError("training data contains no tokens")
        return cls(order, dict(counts), vocabulary)

    def predict(self, prefix: str, limit: int = 10) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        words = tokenize(prefix)
        for n in range(min(self.order - 1, len(words)), -1, -1):
            context = tuple(words[-n:]) if n else ()
            options = self.counts.get(context)
            if options:
                total = sum(options.values())
                ranked = sorted(options.items(), key=lambda item: (-item[1], item[0]))
                return [(word, count / total) for word, count in ranked[:limit]]
        return []

    def to_dict(self, metadata: dict | None = None) -> dict:
        payload = {
            "format": 1,
            "order": self.order,
            "counts": [[list(context), dict(counter)] for context, counter in self.counts.items()],
        }
        if metadata is not None:
            payload["model_version"] = metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "NGramModel":
        if payload.get("format") != 1:
            raise ValueError("unsupported model format")
        counts = {tuple(context): Counter(next_words) for context, next_words in payload["counts"]}
        vocabulary = {word for options in counts.values() for word in options if word not in {"<s>", "</s>"}}
        return cls(int(payload["order"]), counts, vocabulary)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        output = Path(path)
        if output.exists():
            raise FileExistsError(f"model artifact already exists; choose a new versioned path: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(metadata), ensure_ascii=False), encoding="utf-8")
        temporary.replace(output)

    @classmethod
    def load(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_continuation(speeches: list[str], order: int = 5) -> dict[str, float | int]:
    """Train on all but the final speech; score each token of that speech causally."""
    if len(speeches) < 2:
        raise ValueError("need at least two speeches for speech-level holdout evaluation")
    model = NGramModel.train(speeches[:-1], order=order)
    words = tokenize(speeches[-1])
    if not words:
        raise ValueError("held-out speech contains no tokens")
    log_loss = 0.0
    correct = 0
    scored = 0
    prefix: list[str] = []
    for target in words:
        predictions = model.predict(" ".join(prefix), limit=max(10, len(model.vocabulary)))
        probabilities = dict(predictions)
        probability = probabilities.get(target, 0.0)
        log_loss -= math.log(max(probability, 1e-12))
        correct += bool(predictions and predictions[0][0] == target)
        scored += 1
        prefix.append(target)
    return {"tokens": scored, "log_loss": log_loss / scored, "top1_accuracy": correct / scored}
