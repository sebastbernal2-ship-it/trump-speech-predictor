"""Local-only training and inference utilities for SpeechPlan-H."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
import math
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .forecast import load_records
from .segments import segment_record
from .hybrid_nn import SpeechPlanConfig, SpeechPlanHybrid
from .model import tokenize

SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>", "<question-empty>", "<evidence-empty>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID, QUESTION_EMPTY_ID, EVIDENCE_EMPTY_ID = range(6)


def _validate_evidence_dates(documents: Sequence[dict], cutoff: date, subject: str) -> None:
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError(f"invalid evidence record for {subject}")
        if not (document.get("context_id") or document.get("url") or document.get("source_url")):
            raise ValueError(f"evidence has no source identifier or URL for {subject}")
        published = document.get("published_at", "")
        try:
            publication_date = date.fromisoformat(published[:10])
        except (TypeError, ValueError):
            raise ValueError(f"evidence is undated or has an invalid date for {subject}") from None
        if publication_date >= cutoff:
            raise ValueError(f"evidence is not known before the prediction date for {subject}")


def build_vocabulary(
    texts: Sequence[str], min_frequency: int = 2, max_size: int = 20_000,
    required_tokens: Sequence[str] = (),
) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    required = set(required_tokens) - set(SPECIAL)
    capacity = max(0, max_size - len(SPECIAL))
    forced = sorted(required, key=lambda word: (-counts[word], word))[:capacity]
    ordinary = [
        word for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_frequency and word not in SPECIAL and word not in required
    ][: max(0, capacity - len(forced))]
    if not ordinary and not forced:
        raise ValueError("training text has no tokens meeting the vocabulary threshold")
    return [*SPECIAL, *forced, *ordinary]


def chronological_splits(records: list[dict], train_fraction: float = 0.8, validation_fraction: float = 0.1):
    """Split by whole calendar dates so one event date cannot straddle splits."""
    dates = sorted({record["date"] for record in records})
    if len(dates) < 3:
        raise ValueError("need records from at least three distinct dates")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * train_fraction)))
    validation_end = min(len(dates) - 1, max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction))))
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    return tuple(
        [record for record in records if record["date"] in date_set]
        for date_set in (train_dates, validation_dates, test_dates)
    )


class ContinuationChunks(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        vocabulary: Sequence[str],
        context_length: int,
        target_length: int,
        stride: int,
        question_length: int = 64,
        evidence_length: int = 128,
    ):
        self.lookup = {token: index for index, token in enumerate(vocabulary)}
        self.context_length = context_length
        self.target_length = target_length
        self.question_length = question_length
        self.evidence_length = evidence_length
        self.sequences = []
        self.samples = []
        for record in records:
            ids = [self.lookup.get(token, UNK_ID) for token in tokenize(record["text"])]
            if len(ids) < target_length:
                continue
            sequence_index = len(self.sequences)
            self.sequences.append(ids)
            self.samples.extend(
                (sequence_index, start)
                for start in range(0, len(ids) - target_length + 1, stride)
            )
        if not self.samples:
            raise ValueError("split has no complete continuation chunks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_index, start = self.samples[index]
        sequence = self.sequences[sequence_index]
        previous = sequence[max(0, start - self.context_length + 1) : start]
        context = ([BOS_ID] + previous)[-self.context_length :]
        context = [PAD_ID] * (self.context_length - len(context)) + context
        target = sequence[start : start + self.target_length]
        question = [QUESTION_EMPTY_ID] + [PAD_ID] * (self.question_length - 1)
        evidence = [EVIDENCE_EMPTY_ID] + [PAD_ID] * (self.evidence_length - 1)
        return tuple(torch.tensor(values, dtype=torch.long) for values in (context, question, evidence, target))


class QuestionAnswerChunks(Dataset):
    """Full target windows from explicit question-to-president turns."""

    def __init__(
        self, records: Sequence[dict], vocabulary: Sequence[str], context_length: int,
        question_length: int, evidence_length: int, target_length: int, stride: int,
        evidence_by_question: dict[str, Sequence[dict]] | None = None,
    ):
        lookup = {token: index for index, token in enumerate(vocabulary)}
        self.samples = []
        self.pair_count = 0
        self.evidence_pair_count = 0
        self.skipped_short_pairs = 0
        evidence_by_question = evidence_by_question or {}
        for record in records:
            segments = segment_record(record)
            questions = {
                segment["segment_id"]: (index, segment)
                for index, segment in enumerate(segments)
                if segment["segment_type"] == "question"
            }
            answers: dict[str, list[dict]] = {}
            for segment in segments:
                question_id = segment.get("response_to")
                if segment["segment_type"] == "presidential_answer" and question_id in questions:
                    answers.setdefault(question_id, []).append(segment)

            tokenized_segments = [tokenize(segment["utterance_text"]) for segment in segments]
            for question_id, answer_segments in answers.items():
                question_index, question = questions[question_id]
                answer_tokens = [token for segment in answer_segments for token in tokenize(segment["utterance_text"])]
                starts = range(0, len(answer_tokens) - target_length + 1, stride)
                starts = list(starts)
                if not starts:
                    self.skipped_short_pairs += 1
                    continue
                self.pair_count += 1

                prior_tokens = [token for part in tokenized_segments[:question_index] for token in part]
                question_tokens = tokenize(question["utterance_text"])[:question_length]
                question_ids = [lookup.get(token, UNK_ID) for token in question_tokens]
                question_ids = question_ids + [PAD_ID] * (question_length - len(question_ids))

                evidence_docs = evidence_by_question.get(question_id, ())
                if evidence_docs:
                    self.evidence_pair_count += 1
                evidence_tokens = []
                _validate_evidence_dates(evidence_docs, date.fromisoformat(record["date"]), question_id)
                for document in evidence_docs:
                    evidence_tokens.extend(tokenize(f"{document.get('title', '')} {document.get('abstract', '')}"))
                evidence_ids = [lookup.get(token, UNK_ID) for token in evidence_tokens[:evidence_length]]
                if not evidence_ids:
                    evidence_ids = [EVIDENCE_EMPTY_ID]
                evidence_ids += [PAD_ID] * (evidence_length - len(evidence_ids))

                for start in starts:
                    previous = (prior_tokens + answer_tokens[:start])[-(context_length - 1):]
                    prefix = ([BOS_ID] + [lookup.get(token, UNK_ID) for token in previous])[-context_length:]
                    prefix = [PAD_ID] * (context_length - len(prefix)) + prefix
                    target = [lookup.get(token, UNK_ID) for token in answer_tokens[start : start + target_length]]
                    self.samples.append(tuple(
                        torch.tensor(values, dtype=torch.long)
                        for values in (prefix, question_ids, evidence_ids, target)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def load_evidence_pairs(path: str | Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    pairs = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_segment_id")
            documents = row.get("documents")
            if (
                not isinstance(question_id, str) or not question_id
                or not isinstance(documents, list) or any(not isinstance(document, dict) for document in documents)
            ):
                raise ValueError(f"invalid evidence-pair row at line {line_number}")
            if question_id in pairs:
                raise ValueError(f"duplicate evidence mapping for question {question_id}")
            pairs[question_id] = documents
    return pairs


def _loader(dataset, batch_size: int, shuffle: bool, device: torch.device, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


def _evaluate(model, loader, device, condition: str = "combined"):
    model.eval()
    totals = Counter()
    examples = 0
    with torch.inference_mode():
        for prefix, question, evidence, target in loader:
            prefix, question, evidence, target = (
                value.to(device) for value in (prefix, question, evidence, target)
            )
            if condition == "question_only":
                prefix = torch.full_like(prefix, PAD_ID)
                prefix[:, -1] = BOS_ID
                evidence = None
            elif condition == "transcript_only":
                question = None
                evidence = None
            elif condition == "combined_without_evidence":
                evidence = None
            elif condition != "combined":
                raise ValueError(f"unknown evaluation condition: {condition}")
            result = model(prefix, target, question, evidence)
            count = target.shape[0]
            examples += count
            for key in ("loss", "token_loss", "ncm_loss", "vq_loss", "token_accuracy"):
                totals[key] += float(result[key]) * count
    return {key: value / max(1, examples) for key, value in totals.items()}


def _save_checkpoint(path: Path, model: SpeechPlanHybrid, vocabulary: list[str], metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint(vocabulary)
    checkpoint["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint["metadata"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_hybrid(
    input_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    stride: int = 32,
    min_frequency: int = 2,
    max_vocabulary: int = 20_000,
    width: int = 128,
    layers: int = 4,
    heads: int = 4,
    codebook_size: int = 128,
    context_length: int = 128,
    target_length: int = 32,
    question_length: int = 64,
    evidence_length: int = 128,
    device_name: str = "auto",
    max_steps: int = 0,
    seed: int = 17,
    evidence_pairs_path: str | Path | None = None,
):
    if epochs < 1 or batch_size < 1 or stride < 1 or max_steps < 0:
        raise ValueError("epochs, batch size, and stride must be positive; max_steps cannot be negative")
    if min(context_length, target_length, question_length, evidence_length) < 1:
        raise ValueError("all input and target lengths must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    records = load_records(input_path)
    train_records, validation_records, test_records = chronological_splits(records)
    evidence_by_question = load_evidence_pairs(evidence_pairs_path)
    question_dates = {
        segment["segment_id"]: record["date"]
        for record in records for segment in segment_record(record)
        if segment["segment_type"] == "question"
    }
    for question_id, documents in evidence_by_question.items():
        if question_id not in question_dates:
            raise ValueError(f"evidence mapping references unknown question {question_id}")
        _validate_evidence_dates(documents, date.fromisoformat(question_dates[question_id]), question_id)
    train_question_ids = {
        segment["segment_id"]
        for record in train_records
        for segment in segment_record(record)
        if segment["segment_type"] == "question"
    }
    train_evidence = {
        question_id: documents for question_id, documents in evidence_by_question.items()
        if question_id in train_question_ids
    }
    training_texts = [record["text"] for record in train_records]
    training_evidence_texts = [
        f"{document.get('title', '')} {document.get('abstract', '')}"
        for documents in train_evidence.values() for document in documents
    ]
    training_texts.extend(training_evidence_texts)
    evidence_vocabulary = [token for text in training_evidence_texts for token in tokenize(text)]
    vocabulary = build_vocabulary(
        training_texts, min_frequency, max_vocabulary, required_tokens=evidence_vocabulary
    )
    shared = (vocabulary, context_length, target_length, stride, question_length, evidence_length)
    train_data = ContinuationChunks(train_records, *shared)
    validation_data = ContinuationChunks(validation_records, *shared)
    test_data = ContinuationChunks(test_records, *shared)
    train_questions = QuestionAnswerChunks(
        train_records, vocabulary, context_length, question_length, evidence_length, target_length, stride,
        train_evidence,
    )
    validation_questions = QuestionAnswerChunks(
        validation_records, vocabulary, context_length, question_length, evidence_length, target_length, stride,
        evidence_by_question,
    )
    test_questions = QuestionAnswerChunks(
        test_records, vocabulary, context_length, question_length, evidence_length, target_length, stride,
        evidence_by_question,
    )
    mixed = ConcatDataset((train_data, train_questions))
    if len(train_questions):
        weights = [1 / len(train_data)] * len(train_data) + [1 / len(train_questions)] * len(train_questions)
        sampler = WeightedRandomSampler(weights, 2 * max(len(train_data), len(train_questions)), replacement=True)
        train_loader = _loader(mixed, batch_size, False, device, sampler)
    else:
        train_loader = _loader(mixed, batch_size, True, device)
    validation_loader = _loader(validation_data, batch_size, False, device)
    test_loader = _loader(test_data, batch_size, False, device)
    validation_question_loader = _loader(validation_questions, batch_size, False, device)
    test_question_loader = _loader(test_questions, batch_size, False, device)

    config = SpeechPlanConfig(
        vocab_size=len(vocabulary), pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        context_length=context_length, target_length=target_length,
        question_length=question_length, evidence_length=evidence_length,
        question_empty_id=QUESTION_EMPTY_ID, evidence_empty_id=EVIDENCE_EMPTY_ID,
        width=width, layers=layers, heads=heads, codebook_size=codebook_size,
        levels=tuple(level for level in sorted({1, 4, 16, target_length}) if target_length % level == 0),
    )
    model = SpeechPlanHybrid(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    print(
        f"SpeechPlan-H initialized from scratch: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"device={device}; continuation chunks={len(train_data)}/{len(validation_data)}/{len(test_data)}; "
        f"question-answer chunks={len(train_questions)}/{len(validation_questions)}/{len(test_questions)}"
    )

    best_validation = math.inf
    output_path = Path(output_path)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = Counter()
        examples = 0
        for step, (prefix, question, evidence, target) in enumerate(train_loader, 1):
            if max_steps and step > max_steps:
                break
            prefix, question, evidence, target = (
                value.to(device) for value in (prefix, question, evidence, target)
            )
            optimizer.zero_grad(set_to_none=True)
            result = model(prefix, target, question, evidence)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = target.shape[0]
            examples += count
            for key in ("loss", "token_loss", "ncm_loss", "vq_loss", "token_accuracy"):
                totals[key] += float(result[key].detach()) * count
        training = {key: value / max(1, examples) for key, value in totals.items()}
        validation = _evaluate(model, validation_loader, device)
        validation_qa = _evaluate(model, validation_question_loader, device) if len(validation_questions) else {}
        validation_score = (
            (validation["token_loss"] + validation_qa["token_loss"]) / 2
            if validation_qa else validation["loss"]
        )
        print(
            f"epoch {epoch}: train_loss={training['loss']:.4f} "
            f"token={training['token_loss']:.4f} ncm={training['ncm_loss']:.4f} "
            f"vq={training['vq_loss']:.4f} train_top1={training['token_accuracy']:.4f} "
            f"validation_continuation_loss={validation['loss']:.4f} "
            f"validation_qa_loss={validation_qa.get('token_loss', float('nan')):.4f} "
            f"validation_qa_top1={validation_qa.get('token_accuracy', float('nan')):.4f}"
        )
        if validation_score < best_validation:
            best_validation = validation_score
            _save_checkpoint(
                output_path, model, vocabulary,
                {
                    "epoch": epoch,
                    "best_validation_loss": best_validation,
                    "validation_continuation_loss": validation["loss"],
                    "validation_qa_token_loss": validation_qa.get("token_loss"),
                    "question_answer_chunks": {
                        "train": len(train_questions),
                        "validation": len(validation_questions),
                        "test": len(test_questions),
                        "evidence_pairs_train": train_questions.evidence_pair_count,
                        "evidence_pairs_validation": validation_questions.evidence_pair_count,
                        "evidence_pairs_test": test_questions.evidence_pair_count,
                    },
                    "records": {
                        "train": len(train_records),
                        "validation": len(validation_records),
                        "test": len(test_records),
                    },
                    "date_ranges": {
                        "train": [train_records[0]["date"], train_records[-1]["date"]],
                        "validation": [validation_records[0]["date"], validation_records[-1]["date"]],
                        "test": [test_records[0]["date"], test_records[-1]["date"]],
                    },
                    "seed": seed,
                    "input_path": str(input_path),
                    "split": "chronological whole-date groups",
                },
            )

    best_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model.to(device)
    test = _evaluate(model, test_loader, device)
    test_qa = {
        condition: _evaluate(model, test_question_loader, device, condition)
        for condition in ("combined", "combined_without_evidence", "transcript_only", "question_only")
    } if len(test_questions) else {}
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["test_continuation_loss"] = test["loss"]
    checkpoint["metadata"]["test_continuation_top1_accuracy"] = test["token_accuracy"]
    checkpoint["metadata"]["test_qa_combined"] = test_qa
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    print(
        f"held-out chronological continuation loss: {test['loss']:.4f}; "
        f"top1={test['token_accuracy']:.4f}"
    )
    for condition, metrics in test_qa.items():
        print(
            f"held-out Q&A {condition}: token_loss={metrics['token_loss']:.4f}; "
            f"top1={metrics['token_accuracy']:.4f}"
        )
    return {
        "validation_loss": best_validation,
        "test_loss": test["loss"],
        "test_top1_accuracy": test["token_accuracy"],
        "test_qa": test_qa,
        "output": str(output_path),
    }


def load_hybrid(path: str | Path, device_name: str = "auto"):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {1, 2}:
        raise ValueError("unsupported SpeechPlan-H checkpoint format")
    config_values = checkpoint["config"]
    config_values["levels"] = tuple(config_values["levels"])
    config = SpeechPlanConfig(**config_values)
    model = SpeechPlanHybrid(config)
    if checkpoint_format == 1:
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        allowed_missing = (
            "question_position.", "evidence_position.", "channel_embeddings.",
            "question_summary_projection.", "evidence_summary_projection.",
        )
        if incompatible.unexpected_keys or any(
            not key.startswith(allowed_missing) for key in incompatible.missing_keys
        ):
            raise ValueError("legacy SpeechPlan-H checkpoint has incompatible weights")
        model.question_channel_enabled = False
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint["vocabulary"], device, checkpoint.get("metadata", {})


def generate_hybrid(
    checkpoint_path: str | Path,
    prefix_text: str,
    *,
    question_text: str = "",
    evidence_records: Sequence[dict] = (),
    as_of: str | None = None,
    max_new_tokens: int = 32,
    device_name: str = "auto",
) -> str:
    model, vocabulary, device, metadata = load_hybrid(checkpoint_path, device_name)
    lookup = {token: index for index, token in enumerate(vocabulary)}
    tokens = [lookup.get(token, UNK_ID) for token in tokenize(prefix_text)]
    context = ([BOS_ID] + tokens)[-model.config.context_length :]
    context = [PAD_ID] * (model.config.context_length - len(context)) + context
    if question_text and not model.question_channel_enabled:
        raise ValueError("this checkpoint predates question conditioning")
    question_tokens = [lookup.get(token, UNK_ID) for token in tokenize(question_text)[:model.config.question_length]]
    if not question_tokens:
        question_tokens = [model.config.question_empty_id]
    question_tokens += [PAD_ID] * (model.config.question_length - len(question_tokens))

    evidence_text = ""
    if evidence_records:
        if not as_of:
            raise ValueError("as_of is required when supplying external evidence")
        cutoff = date.fromisoformat(as_of)
        _validate_evidence_dates(evidence_records, cutoff, "prediction cutoff")
        qa_metadata = metadata.get("question_answer_chunks", {})
        if not qa_metadata.get("evidence_pairs_train", 0):
            raise ValueError("checkpoint was not trained with dated evidence pairs")
        evidence_text = " ".join(
            f"{document.get('title', '')} {document.get('abstract', '')}" for document in evidence_records
        )
    evidence_tokens = [lookup.get(token, UNK_ID) for token in tokenize(evidence_text)[:model.config.evidence_length]]
    if not evidence_tokens:
        evidence_tokens = [model.config.evidence_empty_id]
    evidence_tokens += [PAD_ID] * (model.config.evidence_length - len(evidence_tokens))

    ids = torch.tensor([context], dtype=torch.long, device=device)
    question_ids = torch.tensor([question_tokens], dtype=torch.long, device=device)
    evidence_ids = torch.tensor([evidence_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        ids, max_new_tokens=max_new_tokens, question_ids=question_ids, evidence_ids=evidence_ids
    )
    words = [vocabulary[index] for index in generated if index not in {PAD_ID, BOS_ID, EOS_ID}]
    closing = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
    text = ""
    for word in words:
        text += word if text and word in closing else (" " if text else "") + word
    return text
