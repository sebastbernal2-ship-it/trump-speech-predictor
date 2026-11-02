# SpeechPlan-H architecture flowchart

This page shows the model that was trained on the long-speech cohort. It is a small model trained from scratch: its decoder uses GPT-2 architecture, but it does **not** load GPT-2 weights or Stanford NCM weights.

## Prediction: observed text to next-token continuation

```mermaid
flowchart TD
    A["Observed transcript prefix<br/>words and punctuation"] --> B["Tokenizer<br/>lowercase word/punctuation tokens"]
    B --> C["Vocabulary lookup<br/>up to 20,000 IDs + padding"]

    C --> D["Token embeddings + positions"]
    D --> E["Context Transformer encoder<br/>2 layers"]
    E --> F["Planner summary<br/>last prefix state + optional channel summaries"]

    Q["Optional question text"] --> QE["Question tokens<br/>same context encoder + question positions"]
    R["Optional dated evidence"] --> RE["Evidence tokens<br/>same context encoder + evidence positions"]
    QE --> AUX["Auxiliary summary + token memory"]
    RE --> AUX
    AUX --> F

    F --> P["NCM-inspired hierarchical planner"]
    P --> L1["Level 1<br/>1 coarse code"]
    L1 --> L4["Level 2<br/>4 codes"]
    L4 --> L16["Level 3<br/>16 codes"]
    L16 --> L32["Level 4<br/>32 fine codes"]
    L1 --> M["Code embeddings concatenated<br/>53 plan-memory slots"]
    L4 --> M
    L16 --> M
    L32 --> M

    C --> DECIN["Decoder input begins with<br/>prefix IDs + BOS"]
    PREV["Previously generated tokens"] --> DECIN
    DECIN --> DEC["GPT-2-style causal decoder<br/>randomly initialized; cross-attends to memory"]
    M --> DEC
    AUX --> DEC
    DEC --> LOGITS["Next-token scores"]
    LOGITS --> PICK["Select/sample next token"]
    PICK --> STOP{"EOS or 32-token segment complete?"}
    STOP -- No --> PREV
    STOP -- Yes --> OUT["Decode token IDs to continuation text"]

    classDef input fill:#e8f1ff,stroke:#3565a8,color:#111;
    classDef latent fill:#fff1d6,stroke:#b77a00,color:#111;
    classDef output fill:#e4f5e8,stroke:#37804a,color:#111;
    class A,B,C,Q,R input;
    class P,L1,L4,L16,L32,M,F,E,QE,RE,AUX,D latent;
    class LOGITS,PICK,OUT output;
```

The plan is computed from the prefix and held fixed while the decoder generates one segment. Each of the four levels has a 128-entry codebook. Within a level, slot predictions are parallel; each finer level is conditioned on the coarser level. The 53 plan slots are learned latent vectors, not 53 words or named concepts. Training uses soft code distributions so gradients can flow; inference selects the highest-scoring code at each slot.

When a question or evidence input is absent, the model receives an empty-channel sentinel. In the long-speech run, the question and evidence channels contained no real question or evidence.

## Training: how the planner learns what codes to predict

The future target is available only as a training label. It is not an input to the planner at inference.

```mermaid
flowchart LR
    PREFIX["Observed prefix tokens"] --> CTX["Context encoder"]
    CTX --> SUMMARY["Prefix summary"]
    SUMMARY --> PLAN["Planner predicts code logits<br/>at 1 / 4 / 16 / 32 slots"]

    TARGET["Gold next 32 tokens<br/>(training only)"] --> TENC["Target Transformer encoder<br/>2 layers"]
    TENC --> HIDDEN["Future hidden vectors"]
    HIDDEN --> VQ["Residual vector quantizer<br/>pool, nearest code, subtract residual"]
    VQ --> CODES["Gold code IDs<br/>at 1 / 4 / 16 / 32 slots"]

    PLAN --> CODELOSS["Code prediction cross-entropy"]
    CODES --> CODELOSS
    VQ --> VQLOSS["Codebook / commitment loss"]

    PREFIX --> DECIN["Prefix + BOS + shifted gold target"]
    TARGET --> DECLOSS["Gold next-token loss"]
    DECIN --> DEC["Causal decoder<br/>cross-attends to predicted plan"]
    PLAN --> DECMEM["Predicted plan embeddings"]
    DECMEM --> DEC
    DEC --> DECLOSS

    CODELOSS --> TOTAL["Total training loss"]
    VQLOSS --> TOTAL
    DECLOSS --> TOTAL
```

The implemented objective is `token loss + 0.25 × code-prediction loss + 0.05 × vector-quantization loss`. For the trained speech model, the vocabulary cap is 20,000, context length is 128 tokens, target length is 32 tokens, and the decoder has 4 layers, width 128, and 4 attention heads. The target encoder and quantizer create supervision; at prediction time, the planner must predict the codes from the observed prefix alone.

## Words, phrases, ideas, and “concepts”

| Level | What the implementation actually has |
|---|---|
| Words | A custom word-and-punctuation tokenizer maps text to vocabulary IDs. |
| Phrases | No separate phrase detector or phrase-planning module. The decoder learns token sequences and may reproduce phrase patterns implicitly. |
| Ideas | No human-labeled idea classifier or explicit idea-prediction output. |
| Concepts | The planner predicts discrete residual VQ code IDs at four time scales. They are learned representations, not readable concept labels, and a code is not guaranteed to represent one idea. |
| Output | A distribution over the next vocabulary token, followed by autoregressive token generation. |

So the actual path is **words → token IDs → learned hidden states and opaque multi-scale codes → next-word probabilities**. “Words → phrases → ideas → concepts” would suggest explicit semantic stages that this implementation does not contain. Adding a real idea layer would require reviewed idea labels, an explicit prediction head, and its own evaluation.

## Separate pretrained NCM + LLM comparison baseline

This is a different pipeline from SpeechPlan-H. It does not share weights with, or train jointly with, the model above.

```mermaid
flowchart LR
    PREFIX["Observed transcript prefix"] --> GPT["Pretrained GPT-2<br/>prefix-conditioned"]
    GPT --> GP["GPT-2 candidate continuations"]

    NCM["Released NCM LM1B-small<br/>unconditional; no prefix"] --> NP["NCM candidate continuations"]

    GP --> UNION["Union candidate pool"]
    NP --> UNION
    UNION --> GS["GPT-2 scores each candidate<br/>given the prefix"]
    UNION --> NS["NCM scores each candidate<br/>without the prefix"]
    PREFIX --> GS
    GS --> FUSE["Weighted candidate score<br/>provisional default weights"]
    NS --> FUSE
    FUSE --> MASS["Normalize over this finite pool<br/>aggregate by first token"]
```

This is proposal generation plus reranking, not a shared NCM-to-LLM neural network. NCM code likelihood and GPT-2 token likelihood use different scales; the default equal weights are not calibrated. The resulting first-token mass is only normalized over generated candidates, not over the full vocabulary.
