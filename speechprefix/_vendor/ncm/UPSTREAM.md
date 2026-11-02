# Vendored NCM model components

- Upstream: https://github.com/HazyResearch/ncm
- Pinned commit: `c70274273f489abc73832b2b98584a8483b6b203`
- Upstream license: MIT; see `LICENSE`.
- Included: official model, VQ-VAE, quantizer, tokenizer, and small utility modules needed for inference and fine-tuning.
- Excluded: upstream evaluation launchers, pickle-based evaluation data, and subprocess orchestration.
- Local safety change: `tokenization/vqvae_tokenizer.py` uses `torch.load(..., weights_only=True)` rather than unrestricted pickle loading.
- Local inference adaptation: the LM1B BERT setup configures `[CLS]` as EOS and decoded samples begin with that sentinel, so the adapter treats an initial `[CLS]` as BOS and trims later EOS markers only.

This vendor snapshot is an implementation reference, not an independent verification of NCM paper claims.
