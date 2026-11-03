"""Adapter for the official HazyResearch NCM LM1B-small checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

from .hf_download import download_snapshot

NCM_REPO = "hazyresearch/ncm-checkpoints"
NCM_WEIGHT_REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
NCM_SOURCE_COMMIT = "c70274273f489abc73832b2b98584a8483b6b203"
BERT_TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


class NCMGenerator:
    """Coarse-to-fine NCM proposal model, initialized from the released LM1B-small weights.

    The released LM1B checkpoint is unconditional: it expects the BERT CLS
    sentinel as its one-token prefix. Its samples can be proposed to an LLM,
    which can rerank them using the actual observed speech prefix. A
    prefix-conditioned NCM needs continuation fine-tuning.
    """

    def __init__(
        self,
        prefix_tokens: int = 1,
        device: str | None = None,
        download_retries: int = 3,
        download_timeout: int = 20,
    ):
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as error:
            raise RuntimeError("Install the project NCM extra to use NCM") from error

        self.torch = torch
        package_root = Path(__file__).resolve().parent
        vendor_root = package_root / "_vendor" / "ncm"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = download_snapshot(
            NCM_REPO,
            retries=download_retries,
            timeout_seconds=download_timeout,
            revision=NCM_WEIGHT_REVISION,
            allow_patterns=[
                "lm1b/tokenizer.pt",
                "lm1b/small/generator.pt",
                "lm1b/small/config.yaml",
            ],
        )
        self.checkpoint_revision = NCM_WEIGHT_REVISION
        self.source_commit = NCM_SOURCE_COMMIT
        self.tokenizer_revision = BERT_TOKENIZER_REVISION
        tokenizer_files = download_snapshot(
            "bert-base-uncased",
            retries=download_retries,
            timeout_seconds=download_timeout,
            revision=BERT_TOKENIZER_REVISION,
            allow_patterns=[
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.txt",
            ],
        )

        config_path = package_root / "configs" / "ncm-lm1b-small.yaml"
        config = OmegaConf.load(config_path)
        config.model.prefix_len = prefix_tokens
        config.tokenizer.pre_tokenizer.model_id = tokenizer_files
        config.tokenizer.vqvae.checkpoint_path = str(Path(checkpoint) / "lm1b" / "tokenizer.pt")

        self.tokenizer = VQVAETokenizer.from_config(config.tokenizer)
        self.model = NextConceptTransformer(vqvae_local=self.tokenizer.vqvae, config=config.model)
        model_path = Path(checkpoint) / "lm1b" / "small" / "generator.pt"
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.level0_probs = state["inference"]["level0_unigram"].float().cpu()
        self.token_unigram = state["inference"]["token_unigram"].double().to(self.device)
        self.frequency_penalty = float(config.generation.frequency_penalty)
        self.frequency_deficit = float(config.generation.frequency_deficit)
        self.adjacency_penalty = [float(v) for v in str(config.generation.adjacency_penalty).split(",")]
        state_dict = state["model"]
        # The attention mask is a deterministic buffer whose shape changes with prefix_len.
        state_dict.pop("attn_mask_for_masking", None)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if set(incompatible.missing_keys) - {"attn_mask_for_masking"} or incompatible.unexpected_keys:
            raise RuntimeError(
                f"NCM checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.to(self.device).eval()
        self.tokenizer.vqvae.to(self.device).eval()
        self.prefix_tokens = int(prefix_tokens)

    def _pick_tokens(self, logits):
        batch, length, vocab = logits.shape
        allowance = (self.token_unigram[:vocab] * length)[None, :]
        counts = self.torch.zeros(batch, vocab, dtype=self.torch.float64, device=logits.device)
        output = self.torch.empty(batch, length, dtype=self.torch.long, device=logits.device)
        history = []
        penalty = self.adjacency_penalty[0] if self.adjacency_penalty else 0.0
        window = int(self.adjacency_penalty[1]) if len(self.adjacency_penalty) > 1 else 1
        decay = self.adjacency_penalty[2] if len(self.adjacency_penalty) > 2 else 1.0
        for step_index in range(length):
            step = logits[:, step_index, :].clone()
            excess = (counts - allowance).clamp_min(0.0)
            step -= self.frequency_penalty * excess
            if self.frequency_deficit > 0:
                step += self.frequency_deficit * ((length - step_index) / length) * (
                    allowance - counts
                ).clamp_min(0.0)
            for offset, prior in enumerate(reversed(history[-window:])):
                step.scatter_add_(
                    1, prior[:, None],
                    self.torch.full((batch, 1), -penalty * decay**offset, dtype=self.torch.float64, device=logits.device),
                )
            selected = step.argmax(-1)
            output[:, step_index] = selected
            history.append(selected)
            counts.scatter_add_(1, selected[:, None], self.torch.ones((batch, 1), dtype=self.torch.float64, device=logits.device))
        return output

    def score_candidates(self, candidates: list[str]) -> list[float]:
        """Score complete 128-token chunks by mean NCM code log-likelihood."""
        scores = []
        tokenizer = self.tokenizer.pre_tokenizer
        pad = tokenizer.pad_token_id
        cls_id = tokenizer.eos_token_id
        context_length = self.tokenizer.vqvae.context_length
        levels = self.tokenizer.vqvae.vq.levels
        for text in candidates:
            ids = tokenizer.encode(text)[:context_length]
            ids += [pad] * (context_length - len(ids))
            target_tokens = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
            prefix_ids = self.torch.full(
                (1, self.prefix_tokens), cls_id, dtype=self.torch.long, device=self.device
            )
            with self.torch.inference_mode():
                codes = self.tokenizer.vqvae.encode(target_tokens)
                ncm_inputs = self.tokenizer.vqvae.vq.idx_to_ncm_input(codes)
                prefix_emb = self.tokenizer.vqvae.encoder.wte(prefix_ids)
                l0_input = self.model.build_l0_query_input(1, prefix_emb.dtype)
                model_input = self.torch.cat([prefix_emb, l0_input, ncm_inputs], dim=1)
                logits = self.model(model_input)[0].float()
                log_probs = self.torch.log_softmax(logits, dim=-1)
                pieces = []
                offset = 0
                for level in levels:
                    target = codes[0, offset : offset + level]
                    pieces.append(log_probs[offset : offset + level].gather(1, target[:, None]).sum())
                    offset += level
                scores.append(float(self.torch.stack(pieces).sum() / sum(levels)))
        return scores

    def generate(self, count: int = 4, temperature: float = 1.0) -> list[str]:
        """Generate decoded NCM continuations; these are proposals, not calibrated probabilities."""
        if count < 1 or temperature <= 0:
            raise ValueError("count and temperature must be positive")
        cls_id = self.tokenizer.pre_tokenizer.eos_token_id
        sample_count = max(count * 4, count + 4)
        prefixes = self.torch.full(
            (sample_count, self.prefix_tokens), cls_id, dtype=self.torch.long, device=self.device
        )

        level0_count = self.tokenizer.vqvae.vq.levels[0]
        level0 = self.torch.multinomial(
            self.level0_probs, sample_count * level0_count, replacement=True
        ).view(sample_count, level0_count).to(self.device)
        with self.torch.inference_mode():
            codes = self.model.generate(
                idx=level0,
                prefix_tokens=prefixes,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
            )
            _, z_levels = self.tokenizer.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            token_logits = self.tokenizer.vqvae.decoder(z_levels[-1]).double()
            token_ids = self._pick_tokens(token_logits).tolist()

        results = []
        eos = self.tokenizer.pre_tokenizer.eos_token_id
        for ids in token_ids:
            # The released BERT setup uses [CLS] as its configured EOS; generated
            # samples begin with that sentinel, so discard the leading BOS first.
            if ids and ids[0] == eos:
                ids = ids[1:]
            if eos in ids:
                ids = ids[:ids.index(eos)]
            text = self.tokenizer.pre_tokenizer.decode(ids, skip_special_tokens=True).strip()
            if text:
                results.append(text)
        return results[:count]
