"""Optional Hugging Face causal-LM baseline and continuation candidate scorer."""

from __future__ import annotations

from pathlib import Path

from .hf_download import download_snapshot
from .hybrid import Candidate


class CausalLM:
    """Small adapter around a Transformers causal language model."""

    def __init__(
        self,
        model_id: str = "openai-community/gpt2",
        device: str | None = None,
        download_retries: int = 3,
        download_timeout: int = 20,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the project ML extra to use the LLM backend") from error

        self.torch = torch
        local_model = Path(model_id)
        snapshot = str(local_model) if local_model.is_dir() else download_snapshot(
            model_id, retries=download_retries, timeout_seconds=download_timeout
        )
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.context_size = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", 1024
        )

    def next_tokens(self, prefix: str, top: int = 10) -> list[tuple[str, float]]:
        """Return model probabilities for the top token pieces (not whole-word mass)."""
        if top < 1:
            raise ValueError("top must be positive")
        encoded = self.tokenizer(
            prefix or " ", return_tensors="pt", truncation=True,
            max_length=self.context_size,
        ).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits[0, -1].float()
            probs = self.torch.softmax(logits, dim=-1)
            values, ids = self.torch.topk(probs, min(top, probs.numel()))
        return [
            (self.tokenizer.decode([int(token_id)]), float(probability))
            for token_id, probability in zip(ids.cpu(), values.cpu())
        ]

    def score_candidates(self, prefix: str, candidates: list[str]) -> list[float]:
        """Score continuations by mean raw LLM token log-likelihood."""
        if not candidates:
            return []
        prompt = self.tokenizer.encode(prefix or " ", add_special_tokens=False)
        prompt = prompt[-self.context_size:]
        scores = []
        for candidate in candidates:
            continuation = self.tokenizer.encode(" " + candidate.lstrip(), add_special_tokens=False)
            if not continuation:
                scores.append(float("-inf"))
                continue
            max_prompt = max(1, self.context_size - len(continuation))
            context = prompt[-max_prompt:]
            sequence = self.torch.tensor([context + continuation], device=self.device)
            context_len = len(context)
            with self.torch.inference_mode():
                logits = self.model(input_ids=sequence).logits[0].float()
                positions = self.torch.arange(
                    context_len - 1, context_len + len(continuation) - 1, device=self.device
                )
                log_probs = self.torch.log_softmax(logits[positions], dim=-1)
                target = sequence[0, context_len:]
                token_scores = log_probs.gather(1, target[:, None]).squeeze(1)
                scores.append(float(token_scores.mean()))
        return scores

    def generate_candidates(
        self,
        prefix: str,
        count: int = 8,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[Candidate]:
        """Sample continuations and score each with mean raw token log-likelihood."""
        if count < 1 or max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation settings")
        encoded = self.tokenizer(
            prefix or " ", return_tensors="pt", truncation=True,
            max_length=max(1, self.context_size - max_new_tokens),
        ).to(self.device)
        prompt_length = encoded.input_ids.shape[1]
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=count,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            results = []
            for sequence in sequences:
                continuation_ids = sequence[prompt_length:]
                eos = self.tokenizer.eos_token_id
                eos_positions = (continuation_ids == eos).nonzero(as_tuple=True)[0]
                if eos_positions.numel():
                    continuation_ids = continuation_ids[: int(eos_positions[0]) + 1]
                if not continuation_ids.numel():
                    continue
                scored = sequence[: prompt_length + continuation_ids.numel()].unsqueeze(0)
                logits = self.model(input_ids=scored).logits[0].float()
                target_positions = self.torch.arange(
                    prompt_length - 1,
                    prompt_length + continuation_ids.numel() - 1,
                    device=self.device,
                )
                log_probs = self.torch.log_softmax(logits[target_positions], dim=-1)
                token_scores = log_probs.gather(1, continuation_ids[:, None]).squeeze(1)
                text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
                if text.strip():
                    results.append(Candidate(text, {"llm": float(token_scores.mean())}))
        return results
