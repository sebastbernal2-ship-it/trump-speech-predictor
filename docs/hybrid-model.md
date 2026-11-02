# Pretrained NCM/LLM candidate-fusion baseline

## What the two models predict

A conventional autoregressive LLM estimates the next-token distribution conditioned on the prefix. It emits tokens sequentially; a token may be a whole word or a word fragment.

The Stanford Next-Concept Modeling paper describes a different generation path: a multiscale residual VQ-VAE maps text to hierarchical discrete codes, then a transformer predicts one code level at a time and predicts the codes within each level in parallel. These learned codes are not automatically human-readable concepts. NCM is therefore not simply an LLM with a larger next-token step, and its output cannot be averaged directly with next-word probabilities.

Reference: Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, arXiv:2610.14162, https://arxiv.org/abs/2610.14162.

## Separate pretrained comparison baseline

This is not the primary custom model; see `speechplan-hybrid.md`. The older candidate-fusion command uses GPT-2 (`openai-community/gpt2`, 124M, MIT) with its standard tokenizer and the official `hazyresearch/ncm-checkpoints` LM1B-small generator, VQ-VAE tokenizer, and BERT-base-uncased. The released NCM generator is unconditional and starts from its one-token CLS sentinel; it samples a generic continuation without seeing the current speech prefix. The baseline unions NCM and GPT-2 samples, scores each candidate under both models, and ranks/marginalizes by first token. The LLM score is prefix-conditioned; the NCM score is an unconditional code likelihood. They use different units and are not calibrated.

The next intended experiment is to make NCM proposals genuinely prefix-conditioned, fine-tune the official generator on clean continuation chunks, and validate that adaptation before comparing it. The original candidate-level fusion design remains:

```text
observed prefix
      ├── autoregressive LLM -> candidate continuations + token log-likelihoods
      └── NCM -> candidate next-segment code hierarchies + code log-likelihoods
                                      │
                         decode / join candidate set
                                      │
               validation-fitted sequence reranker
                                      │
              aggregate candidate mass by next word
```

The candidate join must preserve each model's raw score and provenance. A validation-only calibration layer can learn the contribution of LLM likelihood, NCM likelihood, and optional recency/domain features. Normalize for candidate length and model-specific score scales. Do not average raw logits or claim confidence from uncalibrated scores. Compare the hybrid against each standalone model and a simple n-gram baseline.

This reranker is a hypothesis, not a proven optimal fusion. `speechprefix.hybrid.combine_candidate_mass` and `speechprefix.hybrid_model.LLMNCMHybrid` implement finite-candidate fusion. CLI defaults use equal weights only as a runnable placeholder; do not interpret them as fitted weights. Current component scores are mean log-likelihoods over GPT-2 tokens and NCM codes, respectively. The returned first-token mass is normalized over the generated candidate pool only; it is not a calibrated vocabulary-wide confidence estimate.

If NCM cannot score the same continuation candidates reliably, test a second design: NCM proposes the next segment at coarse-to-fine resolution, and the LLM verbalizes/refines it. Evaluate exact words independently from semantic/segment similarity so a fluent paraphrase does not count as a correct word forecast.

## Separate task tracks

### Prefix-conditioned continuation

Input is the transcript prefix observed so far. Evaluate by replaying held-out speeches causally, scoring before revealing each next token. Also report metrics by prefix length and by document type. Exclude reporter Q&A and written statements from the primary prepared-speech set; keep them as separate robustness sets.

### Before-speech forecast

Input can contain only information known before the event. The target is not an exact transcript reconstruction by default. Start with coarse outcomes such as topic/phrase occurrence and score those separately. To generate candidate passage plans, condition NCM/LLM on event metadata only when the metadata is genuinely available before the speech. Do not pass any text from the target speech.

The current GovInfo records are sparse, title-filtered presidential documents, not a campaign-speech corpus. They can test plumbing and small-domain continuation, but they are insufficient to pretrain either model or support representative full-speech claims.

## Data and model prerequisites

1. The official NCM implementation, license, and LM1B-small checkpoint are identified and integrated. Next, build and validate a speech-domain continuation fine-tune; do not call the current unconditional proposal model prefix-conditioned.
2. The GPT-2 baseline is pinned to `openai-community/gpt2` with its standard tokenizer and MIT model license. Pin prompt and decoding settings in evaluation reports.
3. Establish a clean, rights-reviewed corpus with full text, dates, provenance, type labels, and duplicate detection. Keep whole speeches and dates intact across chronological splits.
4. Fit any fusion weights only on validation data. Lock them before final test evaluation.
5. Report next-token/word log loss and top-k accuracy, continuation coverage, calibration (Brier/reliability), and semantic segment quality as distinct metrics.
6. Compare compute, latency, and sample quality. NCM's paper-reported speed/quality results are for its tested benchmarks, not a guarantee on Trump speech forecasting.

## Decision gate

Do not interpret this pretrained candidate-fusion baseline as SpeechPlan-H. The released checkpoint is generic and unconditional, while the custom model learns its own hierarchical codebooks and text decoder from local data. The current GovInfo records are mixed presidential documents and have not been manually verified as full speeches; clean the corpus before making representative quality claims. Keep pre-speech forecasting completely separate from prefix-conditioned continuation.
