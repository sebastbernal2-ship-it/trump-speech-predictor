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
    question_length: int = 64
    evidence_length: int = 128
    question_empty_id: int = 4
    evidence_empty_id: int = 5

    def __post_init__(self):
        if self.vocab_size < 6 or self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("invalid vocabulary size or transformer dimensions")
        special_ids = (self.pad_id, self.bos_id, self.eos_id, self.question_empty_id, self.evidence_empty_id)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in special_ids):
            raise ValueError("special token IDs must be within the vocabulary")
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("special token IDs must be distinct")
        if min(self.context_length, self.target_length, self.question_length, self.evidence_length, self.layers) < 1:
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
        self.question_position = nn.Embedding(config.question_length, width)
        self.evidence_position = nn.Embedding(config.evidence_length, width)
        self.channel_embeddings = nn.Embedding(2, width)
        self.question_summary_projection = nn.Linear(width, width, bias=False)
        self.evidence_summary_projection = nn.Linear(width, width, bias=False)
        self.question_channel_enabled = True
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

    def _summary(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(~mask, -1).amax(dim=1).clamp_min(0)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

    def _encode_auxiliary(
        self, ids: torch.Tensor | None, *, length: int, empty_id: int, position: nn.Embedding, channel: int,
        batch_size: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ids is None:
            ids = torch.full((batch_size, length), self.config.pad_id, dtype=torch.long, device=device)
            ids[:, 0] = empty_id
        if ids.ndim != 2 or ids.shape[0] != batch_size or ids.shape[1] < 1:
            raise ValueError("auxiliary input must be a nonempty batch of token sequences")
        if ids.shape[1] > length:
            raise ValueError("auxiliary input exceeds its configured length")
        mask = ids.ne(self.config.pad_id)
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            ids = ids.clone()
            ids[empty_rows, 0] = empty_id
            mask = ids.ne(self.config.pad_id)
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = (
            self.token_embedding(ids) + position(positions)[None, :, :]
            + self.channel_embeddings.weight[channel]
        )
        hidden = self.context_encoder(embedded, src_key_padding_mask=~mask)
        return hidden, mask, self._summary(hidden, mask)

    def _context(
        self, prefix_ids: torch.Tensor, question_ids: torch.Tensor | None = None,
        evidence_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_mask = prefix_ids.ne(self.config.pad_id)
        positions = torch.arange(prefix_ids.shape[1], device=prefix_ids.device)
        embedded = self.token_embedding(prefix_ids) + self.context_position(positions)[None, :, :]
        prefix_hidden = self.context_encoder(embedded, src_key_padding_mask=~prefix_mask)
        summary = self._summary(prefix_hidden, prefix_mask)
        if not self.question_channel_enabled:
            empty_memory = prefix_hidden[:, :0]
            empty_mask = prefix_mask[:, :0]
            return summary, empty_memory, empty_mask

        question, question_mask, question_summary = self._encode_auxiliary(
            question_ids, length=self.config.question_length, empty_id=self.config.question_empty_id,
            position=self.question_position, channel=0, batch_size=prefix_ids.shape[0], device=prefix_ids.device,
        )
        evidence, evidence_mask, evidence_summary = self._encode_auxiliary(
            evidence_ids, length=self.config.evidence_length, empty_id=self.config.evidence_empty_id,
            position=self.evidence_position, channel=1, batch_size=prefix_ids.shape[0], device=prefix_ids.device,
        )
        summary = (
            summary + self.question_summary_projection(self._summary(question, question_mask))
            + self.evidence_summary_projection(self._summary(evidence, evidence_mask))
        )
        return summary, torch.cat((question, evidence), dim=1), torch.cat((question_mask, evidence_mask), dim=1)

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

    def forward(
        self, prefix_ids: torch.Tensor, target_ids: torch.Tensor,
        question_ids: torch.Tensor | None = None, evidence_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        summary, auxiliary_memory, auxiliary_mask = self._context(prefix_ids, question_ids, evidence_ids)
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
        encoder_memory = torch.cat((auxiliary_memory, plan_memory), dim=1)
        encoder_mask = torch.cat((auxiliary_mask.long(), plan_mask), dim=1)
        logits = self.decoder(
            input_ids=decoder_input,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_memory,
            encoder_attention_mask=encoder_mask,
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
        question_ids: torch.Tensor | None = None,
        evidence_ids: torch.Tensor | None = None,
    ) -> list[int]:
        if prefix_ids.shape[0] != 1:
            raise ValueError("generation accepts one prefix at a time")
        if temperature <= 0 or top_k < 1:
            raise ValueError("temperature and top_k must be positive")
        summary, auxiliary_memory, auxiliary_mask = self._context(prefix_ids, question_ids, evidence_ids)
        _, plan_memory = self._plan(summary, hard=True)
        encoder_memory = torch.cat((auxiliary_memory, plan_memory), dim=1)
        encoder_mask = torch.cat((auxiliary_mask.long(), torch.ones(
            plan_memory.shape[:2], dtype=torch.long, device=prefix_ids.device
        )), dim=1)
        generated = torch.tensor([[self.config.bos_id]], device=prefix_ids.device)
        maximum = max_new_tokens or self.config.target_length
        for _ in range(min(maximum, self.config.target_length)):
            decoder_input = torch.cat([prefix_ids, generated], dim=1)
            attention_mask = decoder_input.ne(self.config.pad_id).long()
            logits = self.decoder(
                input_ids=decoder_input,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_memory,
                encoder_attention_mask=encoder_mask,
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
            "format": 2,
            "config": asdict(self.config),
            "vocabulary": vocabulary,
            "state_dict": self.state_dict(),
        }
