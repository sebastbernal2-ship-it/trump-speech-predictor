"""SpeechPlan-H: a from-scratch hierarchical-plan + autoregressive LM."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass(frozen=True)
class SpeechPlanConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    context_length: int = 128
    target_length: int = 32
    width: int = 128
    layers: int = 4
    heads: int = 4
    codebook_size: int = 128
    levels: tuple[int, ...] = (1, 4, 16, 32)
    dropout: float = 0.1
    ncm_weight: float = 0.25
    vq_weight: float = 0.05

    def __post_init__(self):
        if self.vocab_size < 4 or self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("invalid vocabulary size or transformer dimensions")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in (self.pad_id, self.bos_id, self.eos_id)):
            raise ValueError("special token IDs must be within the vocabulary")
        if self.context_length < 1 or self.target_length < 1 or self.layers < 1:
            raise ValueError("sequence lengths and layer count must be positive")
        if self.codebook_size < 2 or not self.levels or self.levels[-1] != self.target_length:
            raise ValueError("hierarchy must end at the target length")
        if any(level < 1 or level > self.target_length or self.target_length % level for level in self.levels):
            raise ValueError("each hierarchy level must evenly divide target_length")


class ResidualCodeHierarchy(nn.Module):
    """Residual vector quantization at increasingly fine future-segment scales."""

    def __init__(self, levels: tuple[int, ...], codebook_size: int, width: int):
        super().__init__()
        self.levels = levels
        self.codebooks = nn.ModuleList(nn.Embedding(codebook_size, width) for _ in levels)
        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=0.02)

    def encode(self, hidden: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch, length, width = hidden.shape
        residual = hidden
        codes = []
        losses = []
        for slots, codebook in zip(self.levels, self.codebooks):
            span = length // slots
            pooled = residual.reshape(batch, slots, span, width).mean(dim=2)
            flat = pooled.reshape(-1, width)
            weights = codebook.weight
            distances = (
                flat.square().sum(dim=1, keepdim=True)
                + weights.square().sum(dim=1)[None, :]
                - 2.0 * flat @ weights.t()
            )
            ids = distances.argmin(dim=1).view(batch, slots)
            quantized = codebook(ids)
            losses.append(
                F.mse_loss(quantized, pooled.detach())
                + 0.25 * F.mse_loss(pooled, quantized.detach())
            )
            straight_through = pooled + (quantized - pooled).detach()
            residual = residual - straight_through.repeat_interleave(span, dim=1)
            codes.append(ids)
        return codes, torch.stack(losses).mean()


class SpeechPlanHybrid(nn.Module):
    """NCM-like multiscale planner feeding a GPT-style autoregressive decoder.

    All weights are initialized from scratch. The planner predicts coarse-to-fine
    residual code sequences from the observed prefix. The GPT-style decoder
    cross-attends to those code embeddings while predicting continuation tokens.
    """

    def __init__(self, config: SpeechPlanConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.token_embedding = nn.Embedding(config.vocab_size, width, padding_idx=config.pad_id)
        self.context_position = nn.Embedding(config.context_length, width)
        self.target_position = nn.Embedding(config.target_length, width)
        encoder_layer = nn.TransformerEncoderLayer(
            width, config.heads, 4 * width, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        target_layer = nn.TransformerEncoderLayer(
            width, config.heads, 4 * width, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.target_encoder = nn.TransformerEncoder(target_layer, num_layers=2, enable_nested_tensor=False)
        self.hierarchy = ResidualCodeHierarchy(config.levels, config.codebook_size, width)
        self.level_positions = nn.ParameterList(
            nn.Parameter(torch.randn(1, slots, width) * 0.02) for slots in config.levels
        )
        self.level_embeddings = nn.ParameterList(
            nn.Parameter(torch.randn(1, 1, width) * 0.02) for _ in config.levels
        )
        self.prior_projections = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in config.levels[1:]
        )
        self.code_heads = nn.ModuleList(
            nn.Linear(width, config.codebook_size) for _ in config.levels
        )

        decoder_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.context_length + config.target_length,
            n_ctx=config.context_length + config.target_length,
            n_embd=width,
            n_layer=config.layers,
            n_head=config.heads,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            add_cross_attention=True,
            pad_token_id=config.pad_id,
            bos_token_id=config.bos_id,
            eos_token_id=config.eos_id,
            use_cache=False,
        )
        self.decoder = GPT2LMHeadModel(decoder_config)

    def _context(self, prefix_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = prefix_ids.ne(self.config.pad_id)
        positions = torch.arange(prefix_ids.shape[1], device=prefix_ids.device)
        embedded = self.token_embedding(prefix_ids) + self.context_position(positions)[None, :, :]
        hidden = self.context_encoder(embedded, src_key_padding_mask=~mask)
        last = mask.long().sum(dim=1).sub(1).clamp_min(0)
        summary = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return hidden, summary

    def _plan(
        self, summary: torch.Tensor, hard: bool
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch = summary.shape[0]
        code_logits = []
        memory_levels = []
        previous = None
        for index, (slots, codebook, head) in enumerate(
            zip(self.config.levels, self.hierarchy.codebooks, self.code_heads)
        ):
            query = summary[:, None, :] + self.level_positions[index]
            if previous is not None:
                expanded = F.interpolate(
                    previous.transpose(1, 2), size=slots, mode="nearest"
                ).transpose(1, 2)
                query = query + self.prior_projections[index - 1](expanded)
            logits = head(query)
            probabilities = logits.softmax(dim=-1)
            if hard:
                ids = logits.argmax(dim=-1)
                embeddings = codebook(ids)
            else:
                soft = probabilities @ codebook.weight
                ids = logits.argmax(dim=-1)
                one_hot = F.one_hot(ids, num_classes=self.config.codebook_size).to(probabilities.dtype)
                embeddings = (one_hot - probabilities).detach() @ codebook.weight + soft
            code_logits.append(logits)
            memory_levels.append(embeddings + self.level_embeddings[index])
            previous = embeddings
        return code_logits, torch.cat(memory_levels, dim=1)

    def forward(self, prefix_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        _, summary = self._context(prefix_ids)
        target_positions = torch.arange(target_ids.shape[1], device=target_ids.device)
        target_embedded = self.token_embedding(target_ids) + self.target_position(target_positions)[None, :, :]
        target_hidden = self.target_encoder(target_embedded)
        target_codes, vq_loss = self.hierarchy.encode(target_hidden)
        code_logits, plan_memory = self._plan(summary, hard=False)

        code_losses = [
            F.cross_entropy(logits.reshape(-1, self.config.codebook_size), target.reshape(-1))
            for logits, target in zip(code_logits, target_codes)
        ]
        ncm_loss = torch.stack(code_losses).mean()
        bos = torch.full_like(target_ids[:, :1], self.config.bos_id)
        decoder_tokens = torch.cat([bos, target_ids[:, :-1]], dim=1)
        decoder_input = torch.cat([prefix_ids, decoder_tokens], dim=1)
        decoder_mask = decoder_input.ne(self.config.pad_id).long()
        plan_mask = torch.ones(plan_memory.shape[:2], dtype=decoder_mask.dtype, device=decoder_mask.device)
        logits = self.decoder(
            input_ids=decoder_input,
            attention_mask=decoder_mask,
            encoder_hidden_states=plan_memory,
            encoder_attention_mask=plan_mask,
            logits_to_keep=self.config.target_length,
        ).logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), target_ids.reshape(-1),
            ignore_index=self.config.pad_id,
        )
        valid = target_ids.ne(self.config.pad_id)
        token_accuracy = ((logits.argmax(dim=-1) == target_ids) & valid).sum() / valid.sum().clamp_min(1)
        loss = token_loss + self.config.ncm_weight * ncm_loss + self.config.vq_weight * vq_loss
        return {
            "loss": loss, "token_loss": token_loss, "ncm_loss": ncm_loss,
            "vq_loss": vq_loss, "token_accuracy": token_accuracy,
        }

    @torch.inference_mode()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> list[int]:
        if prefix_ids.shape[0] != 1:
            raise ValueError("generation accepts one prefix at a time")
        if temperature <= 0 or top_k < 1:
            raise ValueError("temperature and top_k must be positive")
        _, summary = self._context(prefix_ids)
        _, plan_memory = self._plan(summary, hard=True)
        generated = torch.tensor([[self.config.bos_id]], device=prefix_ids.device)
        maximum = max_new_tokens or self.config.target_length
        for _ in range(min(maximum, self.config.target_length)):
            decoder_input = torch.cat([prefix_ids, generated], dim=1)
            attention_mask = decoder_input.ne(self.config.pad_id).long()
            logits = self.decoder(
                input_ids=decoder_input,
                attention_mask=attention_mask,
                encoder_hidden_states=plan_memory,
                encoder_attention_mask=torch.ones(plan_memory.shape[:2], device=prefix_ids.device),
                logits_to_keep=1,
            ).logits[0, -1].float() / temperature
            values, indices = torch.topk(logits, min(top_k, logits.numel()))
            next_id = indices[torch.multinomial(values.softmax(dim=-1), 1)]
            generated = torch.cat([generated, next_id.reshape(1, 1)], dim=1)
            if int(next_id) == self.config.eos_id:
                break
        return generated[0, 1:].tolist()

    def checkpoint(self, vocabulary: list[str]) -> dict:
        return {
            "format": 1,
            "config": asdict(self.config),
            "vocabulary": vocabulary,
            "state_dict": self.state_dict(),
        }
