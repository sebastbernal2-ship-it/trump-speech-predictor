"""Separate semantic idea-label and exact-wording evaluation."""

from __future__ import annotations

import re
from typing import Iterable, Sequence


WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
RESPONSE_STATUSES = {
    "target_spoke", "other_speaker_spoke", "no_target_response", "uncertain",
}


def evaluate_response_status(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float | int]:
    """Score whether the exchange outcome was attributed to the right speaker/status."""
    if len(predictions) != len(references):
        raise ValueError("prediction and reference counts must match")
    if any(label not in RESPONSE_STATUSES for label in [*predictions, *references]):
        raise ValueError("unsupported response status")
    accuracy = sum(prediction == reference for prediction, reference in zip(predictions, references))
    labels = set(predictions) | set(references)
    per_class_f1 = []
    for label in labels:
        true_positive = sum(pred == label and ref == label for pred, ref in zip(predictions, references))
        false_positive = sum(pred == label and ref != label for pred, ref in zip(predictions, references))
        false_negative = sum(pred != label and ref == label for pred, ref in zip(predictions, references))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "response_examples_scored": len(references),
        "response_status_accuracy": accuracy / len(references) if references else 0.0,
        "response_status_macro_f1": sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0,
    }


def evaluate_idea_labels(
    predictions: Sequence[Iterable[str]], references: Sequence[Iterable[str]],
) -> dict[str, float | int]:
    """Score only responses with human-supplied idea labels; never use transcript text as labels."""
    if len(predictions) != len(references):
        raise ValueError("prediction and reference counts must match")
    true_positive = false_positive = false_negative = exact = scored = 0
    for predicted, reference in zip(predictions, references):
        if isinstance(predicted, (str, bytes)) or isinstance(reference, (str, bytes)):
            raise ValueError("idea labels must be collections of labels, not strings")
        predicted_set, reference_set = set(predicted), set(reference)
        if any(not isinstance(label, str) or not label for label in predicted_set | reference_set):
            raise ValueError("idea labels must be nonempty strings")
        if not reference_set:
            continue
        true_positive += len(predicted_set & reference_set)
        false_positive += len(predicted_set - reference_set)
        false_negative += len(reference_set - predicted_set)
        exact += predicted_set == reference_set
        scored += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "idea_examples_scored": scored,
        "unlabeled_examples": len(references) - scored,
        "idea_micro_precision": precision,
        "idea_micro_recall": recall,
        "idea_micro_f1": f1,
        "idea_exact_set_match": exact / scored if scored else 0.0,
    }


def _word_tokens(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, 1):
        current = [row]
        for column, right_token in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def evaluate_exact_wording(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float | int]:
    """Report word error rate and exact word-sequence match independently of ideas."""
    if len(predictions) != len(references):
        raise ValueError("prediction and reference counts must match")
    if any(not reference.strip() for reference in references):
        raise ValueError("wording references must be nonempty target-speaker responses")
    if any(not _word_tokens(reference) for reference in references):
        raise ValueError("wording references must contain at least one word")
    errors = 0
    reference_words = 0
    exact = 0
    for prediction, reference in zip(predictions, references):
        predicted_words = _word_tokens(prediction)
        reference_word_sequence = _word_tokens(reference)
        errors += _edit_distance(predicted_words, reference_word_sequence)
        reference_words += len(reference_word_sequence)
        exact += predicted_words == reference_word_sequence
    return {
        "responses_scored": len(references),
        "word_error_rate": errors / reference_words if reference_words else 0.0,
        "exact_word_sequence_match": exact / len(references) if references else 0.0,
        "exact_text_match": sum(
            " ".join(prediction.casefold().split()) == " ".join(reference.casefold().split())
            for prediction, reference in zip(predictions, references)
        ) / len(references) if references else 0.0,
    }
