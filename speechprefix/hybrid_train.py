"""Local-only training and inference utilities for SpeechPlan-H."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .forecast import load_records
from .hybrid_nn import SpeechPlanConfig, SpeechPlanHybrid
from .model import tokenize

SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)


def build_vocabulary(texts: Sequence[str], min_frequency: int = 2, max_size: int = 20_000) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    ordinary = [
        word for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_frequency and word not in SPECIAL
    ][: max(0, max_size - len(SPECIAL))]
    if not ordinary:
        raise ValueError("training text has no tokens meeting the vocabulary threshold")
    return [*SPECIAL, *ordinary]


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
    ):
        self.lookup = {token: index for index, token in enumerate(vocabulary)}
        self.context_length = context_length
        self.target_length = target_length
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
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def _loader(dataset, batch_size: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


def _evaluate(model, loader, device):
    model.eval()
    totals = Counter()
    examples = 0
    with torch.inference_mode():
        for prefix, target in loader:
            result = model(prefix.to(device), target.to(device))
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
    device_name: str = "auto",
    max_steps: int = 0,
    seed: int = 17,
):
    if epochs < 1 or batch_size < 1 or stride < 1 or max_steps < 0:
        raise ValueError("epochs, batch size, and stride must be positive; max_steps cannot be negative")
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
    vocabulary = build_vocabulary(
        [record["text"] for record in train_records], min_frequency, max_vocabulary
    )
    shared = (vocabulary, context_length, target_length, stride)
    train_data = ContinuationChunks(train_records, *shared)
    validation_data = ContinuationChunks(validation_records, *shared)
    test_data = ContinuationChunks(test_records, *shared)
    train_loader = _loader(train_data, batch_size, True, device)
    validation_loader = _loader(validation_data, batch_size, False, device)
    test_loader = _loader(test_data, batch_size, False, device)

    config = SpeechPlanConfig(
        vocab_size=len(vocabulary), pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        context_length=context_length, target_length=target_length,
        width=width, layers=layers, heads=heads, codebook_size=codebook_size,
        levels=tuple(level for level in sorted({1, 4, 16, target_length}) if target_length % level == 0),
    )
    model = SpeechPlanHybrid(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    print(
        f"SpeechPlan-H initialized from scratch: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"device={device}; train/validation/test chunks="
        f"{len(train_data)}/{len(validation_data)}/{len(test_data)}"
    )

    best_validation = math.inf
    output_path = Path(output_path)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = Counter()
        examples = 0
        for step, (prefix, target) in enumerate(train_loader, 1):
            if max_steps and step > max_steps:
                break
            prefix, target = prefix.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(prefix, target)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = target.shape[0]
            examples += count
            for key in ("loss", "token_loss", "ncm_loss", "vq_loss", "token_accuracy"):
                totals[key] += float(result[key].detach()) * count
        training = {key: value / max(1, examples) for key, value in totals.items()}
        validation = _evaluate(model, validation_loader, device)
        print(
            f"epoch {epoch}: train_loss={training['loss']:.4f} "
            f"token={training['token_loss']:.4f} ncm={training['ncm_loss']:.4f} "
            f"vq={training['vq_loss']:.4f} train_top1={training['token_accuracy']:.4f} "
            f"validation_loss={validation['loss']:.4f} validation_top1={validation['token_accuracy']:.4f}"
        )
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            _save_checkpoint(
                output_path, model, vocabulary,
                {
                    "epoch": epoch,
                    "best_validation_loss": best_validation,
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
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["test_loss"] = test["loss"]
    checkpoint["metadata"]["test_top1_accuracy"] = test["token_accuracy"]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    print(
        f"held-out chronological test loss (selected by validation): {test['loss']:.4f}; "
        f"top1={test['token_accuracy']:.4f}"
    )
    return {
        "validation_loss": best_validation,
        "test_loss": test["loss"],
        "test_top1_accuracy": test["token_accuracy"],
        "output": str(output_path),
    }


def load_hybrid(path: str | Path, device_name: str = "auto"):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1:
        raise ValueError("unsupported SpeechPlan-H checkpoint format")
    config_values = checkpoint["config"]
    config_values["levels"] = tuple(config_values["levels"])
    config = SpeechPlanConfig(**config_values)
    model = SpeechPlanHybrid(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint["vocabulary"], device


def generate_hybrid(
    checkpoint_path: str | Path,
    prefix_text: str,
    *,
    max_new_tokens: int = 32,
    device_name: str = "auto",
) -> str:
    model, vocabulary, device = load_hybrid(checkpoint_path, device_name)
    lookup = {token: index for index, token in enumerate(vocabulary)}
    tokens = [lookup.get(token, UNK_ID) for token in tokenize(prefix_text)]
    context = ([BOS_ID] + tokens)[-model.config.context_length :]
    context = [PAD_ID] * (model.config.context_length - len(context)) + context
    ids = torch.tensor([context], dtype=torch.long, device=device)
    generated = model.generate(ids, max_new_tokens=max_new_tokens)
    words = [vocabulary[index] for index in generated if index not in {PAD_ID, BOS_ID, EOS_ID}]
    closing = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
    text = ""
    for word in words:
        text += word if text and word in closing else (" " if text else "") + word
    return text
