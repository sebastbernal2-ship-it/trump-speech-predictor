"""LoRA fine-tune a small local causal LM on transcript -> annotation-graph pairs, one fold held out."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPT = 'Annotate the target speaker\'s transcript as a linked JSON graph.\n\nTranscript:\n{input}\n\nAnnotation JSON:\n'


def compact(target: str) -> str:
    return json.dumps(json.loads(target), ensure_ascii=False, separators=(',', ':'))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--holdout', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--lr', type=float, default=2e-4)
    a = p.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in a.pairs.read_text().splitlines()]
    train = [r for r in rows if r['fold'] != a.holdout]
    test = [r for r in rows if r['fold'] == a.holdout]
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, local_files_only=True).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type='CAUSAL_LM',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    examples = []
    for r in train:
        prompt = tok(PROMPT.format(input=r['input']))['input_ids']
        target = tok(compact(r['target']) + tok.eos_token)['input_ids']
        if len(prompt) + len(target) <= a.max_tokens:
            examples.append((prompt, target))
    print(f'train {len(examples)}/{len(train)} kept, test {len(test)}', flush=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    rng = random.Random(0)
    model.train()
    for epoch in range(a.epochs):
        rng.shuffle(examples); total = 0.0
        for prompt, target in examples:
            ids = torch.tensor([prompt + target]).cuda()
            labels = torch.tensor([[-100] * len(prompt) + target]).cuda()
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad(); total += loss.item()
        print(f'epoch {epoch} loss {total / len(examples):.4f}', flush=True)
    model.eval(); a.output.mkdir(parents=True, exist_ok=True)
    for r in test:
        ids = tok(PROMPT.format(input=r['input']), return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=4096, do_sample=False, repetition_penalty=1.05)
        text = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        (a.output / f"{r['view_id']}.json").write_text(json.dumps({'view_id': r['view_id'], 'raw': text}))
    print('done', len(test))


if __name__ == '__main__':
    main()
