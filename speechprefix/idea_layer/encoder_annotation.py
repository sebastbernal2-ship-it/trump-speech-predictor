"""Offline encoder variant of the extractive annotation pipeline."""
from pathlib import Path

from .classical_annotation import annotate

VERSION = 'local-encoder-extractive-v1'


def build_annotator(model_dir):
    """Load trusted local safetensors weights; never resolve a remote model ID."""
    path = Path(model_dir).resolve(strict=True)
    if not path.is_dir():
        raise ValueError('an existing local encoder directory is required')
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                      use_safetensors=True).eval()
    if model.config.is_decoder or model.config.is_encoder_decoder or model.config.model_type not in {
        'bert', 'roberta', 'distilbert', 'mpnet', 'xlm-roberta', 'albert', 'deberta', 'deberta-v2',
    }:
        raise ValueError('an encoder-only checkpoint is required')

    def similarity(labels):
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(labels), 32):
                items = tokenizer(labels[start:start + 32], return_tensors='pt', padding=True,
                                  truncation=True, max_length=128)
                hidden = model(**items).last_hidden_state
                mask = items['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vectors.extend(torch.nn.functional.normalize(pooled, dim=1).unbind())
        return lambda i, j: torch.dot(vectors[i], vectors[j]).item()

    return lambda packet: annotate(packet, similarity=similarity, threshold=.85, engine=VERSION)
