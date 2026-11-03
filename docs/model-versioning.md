# Model versions and artifact tracking

## Registry

`models/registry.jsonl` is the append-only ledger for trained model artifacts. The `train` and `hybrid-train` commands register each run automatically. Every new entry records:

- a unique run-level `model_version` and architecture identifier;
- the exact input-data SHA-256, optional evidence-data SHA-256, and training configuration;
- the source Git revision, dirty-tree flag, and hash of the installed project source;
- relevant runtime versions, split ranges, and evaluation metrics;
- the output artifact path and SHA-256.

The same provenance travels inside the SpeechPlan-H checkpoint or N-gram model file. New training runs refuse to overwrite an existing or previously registered artifact path; choose a new `--output` for every experiment. To compare model architectures, use distinct architecture identifiers (for example, the current `speechplan-h/format-2` versus a future explicitly hierarchical planner), not just different filenames.

## Legacy artifacts

The registry also inventories the four model files already present locally: the N-gram baseline and three SpeechPlan-H checkpoints. Their artifact hashes and available configurations/metrics are recorded, but they predate this scheme. Their exact training-data hashes, source fingerprints, and run IDs were not captured, so these entries are marked `legacy_incomplete`; current files at the same paths must not be mistaken for the exact historical inputs.

External, untrained comparison backends are identified by immutable Hugging Face commit revisions: GPT-2 `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`; NCM checkpoint `hazyresearch/ncm-checkpoints` at `d342ce8aaea58830eef3b8ad473664673e41f692`; and its BERT tokenizer at `bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`. The vendored NCM source is separately pinned to `c70274273f489abc73832b2b98584a8483b6b203`. CLI runs print the resolved model revisions.

Model weights and transcript data remain local and ignored by Git. The registry tracks artifact identity, not artifact availability: preserve or back up each checkpoint if it must remain loadable. A registry entry whose local artifact is missing is a record of a run, not a recoverable model.

## Semantic planner versions

The explicit theme → section → talking-point planner is a separate future architecture and must receive its own architecture identifier and run entry. Draft plan annotations are not training labels. When the semantic model is trained, record the reviewed-plan dataset hash and keep its idea-prediction metrics separate from continuation and wording metrics.
