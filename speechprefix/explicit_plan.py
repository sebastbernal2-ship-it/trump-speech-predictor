"""From-scratch causal predictor for structured theme/section/talking-point paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

from .hybrid_train import BOS_ID, EOS_ID, PAD_ID, UNK_ID, build_vocabulary
from .model import tokenize
from .model_versions import DEFAULT_REGISTRY, assert_new_model_artifact, create_model_version, register_model_version
from .speech_plans import (
    PLAN_MARKERS, WEAK_PLAN_ANNOTATION_SOURCE, plan_path_examples, validate_speech_plan,
)


@dataclass(frozen=True)
class ExplicitPlanConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    plan_start_id: int
    end_plan_id: int
    context_length: int = 128
    plan_length: int = 96
    width: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 6 or self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("invalid vocabulary size or transformer dimensions")
        ids = (self.pad_id, self.bos_id, self.eos_id, self.plan_start_id, self.end_plan_id)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("special token IDs must be distinct and within the vocabulary")
        if min(self.context_length, self.plan_length, self.layers) < 1:
            raise ValueError("sequence lengths and layer count must be positive")


class ExplicitPlanPathModel(nn.Module):
    """Predict a structured plan path from the observed transcript prefix."""

    def __init__(self, config: ExplicitPlanConfig):
        super().__init__()
        self.config = config
        decoder_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.context_length + config.plan_length,
            n_ctx=config.context_length + config.plan_length,
            n_embd=config.width,
            n_layer=config.layers,
            n_head=config.heads,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            pad_token_id=config.pad_id,
            bos_token_id=config.bos_id,
            eos_token_id=config.eos_id,
            use_cache=False,
        )
        self.decoder = GPT2LMHeadModel(decoder_config)

    def forward(self, context_ids: torch.Tensor, plan_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if context_ids.ndim != 2 or context_ids.shape[1] != self.config.context_length:
            raise ValueError("context_ids must use the configured fixed context length")
        if plan_ids.ndim != 2 or plan_ids.shape[1] != self.config.plan_length:
            raise ValueError("plan_ids must use the configured fixed plan length")
        plan_input = torch.full_like(plan_ids, self.config.pad_id)
        plan_input[:, 0] = self.config.plan_start_id
        plan_input[:, 1:] = plan_ids[:, :-1]
        input_ids = torch.cat((context_ids, plan_input), dim=1)
        attention_mask = input_ids.ne(self.config.pad_id).long()
        logits = self.decoder(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, self.config.context_length :, :]
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            plan_ids.reshape(-1),
            ignore_index=self.config.pad_id,
        )
        valid = plan_ids.ne(self.config.pad_id)
        token_accuracy = ((logits.argmax(dim=-1) == plan_ids) & valid).sum() / valid.sum().clamp_min(1)
        return {"loss": loss, "token_accuracy": token_accuracy, "logits": logits}

    @torch.inference_mode()
    def generate(self, context_ids: torch.Tensor, *, max_new_tokens: int | None = None) -> list[int]:
        if context_ids.shape != (1, self.config.context_length):
            raise ValueError("generation accepts one fixed-length context at a time")
        was_training = self.training
        self.eval()
        try:
            generated = torch.tensor([[self.config.plan_start_id]], dtype=torch.long, device=context_ids.device)
            maximum = max_new_tokens or self.config.plan_length
            for _ in range(min(maximum, self.config.plan_length)):
                input_ids = torch.cat((context_ids, generated), dim=1)
                attention_mask = input_ids.ne(self.config.pad_id).long()
                logits = self.decoder(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
                next_id = int(logits.argmax())
                generated = torch.cat((generated, generated.new_tensor([[next_id]])), dim=1)
                if next_id == self.config.end_plan_id:
                    break
            return generated[0, 1:].tolist()
        finally:
            self.train(was_training)


def _event_records(
    events_path: str | Path, plans_path: str | Path, label_policy: str,
) -> list[tuple[dict, dict]]:
    plans = {}
    with Path(plans_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            plan = json.loads(line)
            event_id = plan.get("event_id") if isinstance(plan, dict) else None
            if not isinstance(event_id, str) or event_id in plans:
                raise ValueError(f"plan event IDs must be present and unique (line {line_number})")
            plans[event_id] = plan
    if not plans:
        raise ValueError("no speech plan labels were supplied")

    events = {}
    with Path(events_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_id") in plans:
                events[event["event_id"]] = event
    missing = sorted(set(plans) - set(events))
    if missing:
        raise ValueError(f"source events missing for reviewed plans: {', '.join(missing[:5])}")

    records = []
    for event_id, plan in plans.items():
        event = events[event_id]
        if label_policy == "reviewed":
            validate_speech_plan(plan, event, require_reviewed=True)
        elif label_policy == "weak-extractive":
            if (
                plan.get("review_status") != "draft"
                or plan.get("annotation_source") != WEAK_PLAN_ANNOTATION_SOURCE
            ):
                raise ValueError("weak-extractive training requires draft labels from the extractive labeler")
            validate_speech_plan(plan, event, require_complete=True)
        else:
            raise ValueError(f"unknown plan label policy: {label_policy}")
        if not isinstance(event.get("event_date"), str):
            raise ValueError(f"source event has no date: {event_id}")
        records.append((event, plan))
    return sorted(records, key=lambda row: (row[0]["event_date"], row[0]["event_id"]))


def _chronological_split(records: list[tuple[dict, dict]]):
    dates = sorted({event["event_date"] for event, _ in records})
    if len(dates) < 3:
        raise ValueError("explicit plan training requires at least three distinct event dates")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * 0.8)))
    validation_end = min(
        len(dates) - 1,
        max(train_end + 1, int(len(dates) * 0.9)),
    )
    groups = [set(dates[:train_end]), set(dates[train_end:validation_end]), set(dates[validation_end:])]
    return tuple(
        [(event, plan) for event, plan in records if event["event_date"] in group]
        for group in groups
    )


def _flatten_examples(
    records: Sequence[tuple[dict, dict]], context_length: int, label_policy: str,
) -> list[dict]:
    return [
        example for event, plan in records
        for example in plan_path_examples(
            event, plan, max_context_tokens=context_length, label_policy=label_policy,
        )
    ]


def _encode_text(text: str, lookup: dict[str, int], length: int) -> list[int]:
    tokens = [lookup.get(token, UNK_ID) for token in tokenize(text)][-length:]
    return [PAD_ID] * (length - len(tokens)) + tokens


def _encode_plan(text: str, lookup: dict[str, int], length: int, end_plan_id: int) -> list[int]:
    tokens = [lookup.get(token, UNK_ID) for token in tokenize(text)]
    tokens = tokens[: max(0, length - 1)]
    if not tokens or tokens[-1] != end_plan_id:
        tokens.append(end_plan_id)
    return tokens + [PAD_ID] * (length - len(tokens))


def _tensor_data(examples: Sequence[dict], vocabulary: list[str], config: ExplicitPlanConfig) -> TensorDataset:
    lookup = {token: index for index, token in enumerate(vocabulary)}
    contexts = [_encode_text(row["prefix_text"], lookup, config.context_length) for row in examples]
    plans = [_encode_plan(row["target"]["text"], lookup, config.plan_length, config.end_plan_id) for row in examples]
    return TensorDataset(torch.tensor(contexts, dtype=torch.long), torch.tensor(plans, dtype=torch.long))


def _evaluate(
    model: ExplicitPlanPathModel, loader: DataLoader, device: torch.device, *, greedy_paths: bool = True,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    tokens = 0
    examples = 0
    exact = 0
    with torch.inference_mode():
        for contexts, targets in loader:
            contexts, targets = contexts.to(device), targets.to(device)
            result = model(contexts, targets)
            count = targets.shape[0]
            valid = targets.ne(model.config.pad_id)
            token_count = int(valid.sum())
            total_loss += float(result["loss"]) * token_count
            correct_tokens += int(((result["logits"].argmax(dim=-1) == targets) & valid).sum())
            tokens += token_count
            examples += count
            if greedy_paths:
                for context, target in zip(contexts, targets):
                    expected = target.tolist()
                    expected = expected[:expected.index(model.config.end_plan_id) + 1] if model.config.end_plan_id in expected else expected
                    predicted = model.generate(
                        context.unsqueeze(0), max_new_tokens=len(expected),
                    )
                    exact += predicted == expected
    return {
        "loss": total_loss / max(tokens, 1),
        "teacher_forced_token_accuracy": correct_tokens / max(tokens, 1),
        "greedy_exact_path_accuracy": exact / max(examples, 1) if greedy_paths else None,
        "examples": examples,
        "tokens": tokens,
    }


def train_explicit_plan_model(
    events_path: str | Path,
    plans_path: str | Path,
    output_path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    epochs: int = 5,
    batch_size: int = 16,
    context_length: int = 128,
    plan_length: int = 96,
    width: int = 128,
    layers: int = 4,
    heads: int = 4,
    min_frequency: int = 2,
    max_vocabulary: int = 20_000,
    max_steps: int = 0,
    seed: int = 17,
    device_name: str = "auto",
    label_policy: str = "reviewed",
    cpu_threads: int | None = None,
) -> dict:
    if epochs < 1 or batch_size < 1 or max_steps < 0:
        raise ValueError("epochs and batch size must be positive; max_steps cannot be negative")
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    output = Path(output_path)
    assert_new_model_artifact(output, registry_path)
    if label_policy not in {"reviewed", "weak-extractive"}:
        raise ValueError("label_policy must be reviewed or weak-extractive")
    records = _event_records(events_path, plans_path, label_policy)
    train_records, validation_records, test_records = _chronological_split(records)
    train_examples = _flatten_examples(train_records, context_length, label_policy)
    validation_examples = _flatten_examples(validation_records, context_length, label_policy)
    test_examples = _flatten_examples(test_records, context_length, label_policy)
    if not train_examples or not validation_examples or not test_examples:
        raise ValueError("each chronological split must contain at least one talking-point example")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    if cpu_threads is not None and device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    train_texts = [row["prefix_text"] for row in train_examples]
    train_texts.extend(row["target"]["text"] for row in train_examples)
    vocabulary = build_vocabulary(
        train_texts, min_frequency, max_vocabulary, required_tokens=PLAN_MARKERS
    )
    lookup = {token: index for index, token in enumerate(vocabulary)}
    if "__plan__" not in lookup or "__endplan__" not in lookup:
        raise ValueError("plan control tokens are missing from the vocabulary")
    config = ExplicitPlanConfig(
        vocab_size=len(vocabulary), pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        plan_start_id=lookup["__plan__"], end_plan_id=lookup["__endplan__"],
        context_length=context_length, plan_length=plan_length,
        width=width, layers=layers, heads=heads,
    )
    model = ExplicitPlanPathModel(config).to(device)
    train_data = _tensor_data(train_examples, vocabulary, config)
    validation_data = _tensor_data(validation_examples, vocabulary, config)
    test_data = _tensor_data(test_examples, vocabulary, config)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    version = create_model_version(
        model_family="speechplan-explicit",
        architecture="plan-path-v1",
        task="in-speech theme-section-talking-point prediction",
        dataset_path=events_path,
        additional_data_path=plans_path,
        config=asdict(config),
        training={
            "epochs": epochs, "batch_size": batch_size, "max_steps_per_epoch": max_steps,
            "seed": seed, "device": str(device), "optimizer": "AdamW",
            "learning_rate": 3e-4, "weight_decay": 0.01,
            "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "label_policy": label_policy,
            "examples": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
            },
            "events": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "date_ranges": {
                name: [rows[0][0]["event_date"], rows[-1][0]["event_date"]]
                for name, rows in zip(("train", "validation", "test"), (train_records, validation_records, test_records))
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_validation = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_tokens = 0
        for step, (contexts, targets) in enumerate(train_loader, start=1):
            if max_steps and step > max_steps:
                break
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(contexts, targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            token_count = int(targets.ne(config.pad_id).sum())
            total_loss += float(result["loss"].detach()) * token_count
            train_tokens += token_count
        validation = _evaluate(model, validation_loader, device, greedy_paths=False)
        print(
            f"explicit-plan epoch {epoch}: train_loss={total_loss / max(1, train_tokens):.4f} "
            f"validation_loss={validation['loss']:.4f}"
        )
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            _save_explicit_checkpoint(output, model, vocabulary, version, {
                "best_validation": validation,
                "model_version": version,
            })

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    test = _evaluate(model, test_loader, device)
    checkpoint["metadata"]["test"] = test
    _write_checkpoint(output, checkpoint)
    register_model_version(
        output, version, registry_path,
        metrics={"validation_loss": best_validation, "test": test},
    )
    return {
        "output": str(output),
        "model_version": version["model_version"],
        "validation_loss": best_validation,
        "test": test,
    }


def _write_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _save_explicit_checkpoint(
    path: Path, model: ExplicitPlanPathModel, vocabulary: list[str], version: dict, metadata: dict,
) -> None:
    _write_checkpoint(path, {
        "format": 1,
        "architecture": "speechplan-explicit/plan-path-v1",
        "config": asdict(model.config),
        "vocabulary": vocabulary,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {**metadata, "model_version": version},
    })


def load_explicit_plan_model(path: str | Path, device_name: str = "auto"):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1 or checkpoint.get("architecture") != "speechplan-explicit/plan-path-v1":
        raise ValueError("unsupported explicit plan checkpoint")
    config = ExplicitPlanConfig(**checkpoint["config"])
    model = ExplicitPlanPathModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint["vocabulary"], device, checkpoint.get("metadata", {})


def predict_plan_path(
    checkpoint_path: str | Path,
    prefix_text: str,
    *,
    device_name: str = "auto",
    max_new_tokens: int | None = None,
) -> dict:
    model, vocabulary, device, metadata = load_explicit_plan_model(checkpoint_path, device_name)
    lookup = {token: index for index, token in enumerate(vocabulary)}
    context = _encode_text(prefix_text, lookup, model.config.context_length)
    context_ids = torch.tensor([context], dtype=torch.long, device=device)
    generated = model.generate(context_ids, max_new_tokens=max_new_tokens)
    tokens = [vocabulary[index] for index in generated if index < len(vocabulary)]
    return {"model_version": metadata.get("model_version", {}).get("model_version"), **parse_plan_path(tokens)}


def parse_plan_path(tokens: Sequence[str]) -> dict[str, str]:
    fields = {"__theme__": "theme", "__section__": "section", "__point__": "point"}
    order = ("theme", "section", "point")
    values = {name: [] for name in order}
    active = None
    seen = []
    complete = False
    for token in tokens:
        if token == "__endplan__":
            complete = True
            break
        if token in fields:
            active = fields[token]
            if active in seen:
                raise ValueError("generated plan path repeats a hierarchy marker")
            seen.append(active)
        elif token == "__plan__":
            raise ValueError("generated plan path contains a nested start marker")
        elif active is not None and token not in {"<pad>", "<unk>"}:
            values[active].append(token)
    output = {name: " ".join(words).strip() for name, words in values.items()}
    if not complete or tuple(seen) != order or not all(output.values()):
        raise ValueError("generated sequence did not contain a complete ordered theme/section/point path")
    return output
