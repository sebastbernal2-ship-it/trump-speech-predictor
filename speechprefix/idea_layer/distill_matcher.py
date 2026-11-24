"""Grouped 5-fold comparison of matchers for distilled-annotator pairs.

Compares, on the 375 judged calibration pairs (judge-a labels merged with
calib-pairs.json, grouped by view_id), using only locally cached models
(HF_HUB_OFFLINE=1, no downloads):

  (a) MiniLM cosine (precomputed ``cosine`` field, all-MiniLM-L6-v2),
      threshold maximising F1 on train folds
  (b) distilroberta-base fine-tuned per fold on the 4-way label
      (same / broader / narrower / different), input
      ``"layer: reference" [SEP] generated``; decision P(same)>=0.5 (strict)
      and 1-P(different)>=0.5 (multi)
  (c) feature classifiers on [cosine, layer one-hot, token Jaccard overlap,
      length ratio, negation mismatch]: logistic regression (L2, numpy) and a
      gradient-boosted stump ensemble (numpy; sklearn/lightgbm not installed).
      Decision threshold 0.5 on predicted probability.

Two binary targets per method:
  strict : positive iff label == same
  multi  : positive iff label in {same, broader, narrower}

Runs on CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_B = Path(
    "/home/vorte/code/trump-speech-predictor/data/source-archives/"
    "speech-content/outcomes/distill-v1"
)
CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
LABELS4 = ["same", "broader", "narrower", "different"]
LAYERS = ["ideas", "key_points", "topics", "opinions", "emotions"]
NEG = {"not", "no", "never", "none", "nothing", "neither", "nor", "cannot", "without"}


def resolve_model(name: str) -> str:
    snaps = sorted((CACHE / f"models--{name.replace('/', '--')}" / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"model {name} not in local cache")
    return str(snaps[0])


def load_pairs(base: Path) -> list[dict]:
    judged: dict[str, str] = {}
    for path in sorted(base.glob("judge-a-*.json")):
        for row in json.loads(path.read_text()):
            judged[row["pair_id"]] = row["label"]
    pairs = []
    for row in json.loads((base / "calib-pairs.json").read_text()):
        if row["pair_id"] not in judged:
            raise SystemExit(f"missing judge label for {row['pair_id']}")
        pairs.append({**row, "label": judged[row["pair_id"]]})
    if len(pairs) != 375:
        raise SystemExit(f"expected 375 pairs, got {len(pairs)}")
    return pairs


def make_folds(groups: list[str], k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Assign whole view_id groups to k folds, balancing row counts greedily."""
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    counts = {g: groups.count(g) for g in uniq}
    fold_of: dict[str, int] = {}
    sizes = [0] * k
    for g in sorted(uniq, key=lambda g: -counts[g]):
        f = min(range(k), key=lambda i: (sizes[i], i))
        fold_of[g] = f
        sizes[f] += counts[g]
    fold_idx = np.array([fold_of[g] for g in groups])
    return [np.where(fold_idx == f)[0] for f in range(k)]


def binary_metrics(y: np.ndarray, p: np.ndarray, layers: list[str]) -> dict:
    def stats(mask):
        yt, yp = y[mask], p[mask]
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        tn = int(np.sum(~yt & ~yp))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / max(1, mask.sum())
        return {
            "n": int(mask.sum()),
            "positives": int(yt.sum()),
            "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        }

    out = {"overall": stats(np.ones_like(y, dtype=bool)), "per_layer": {}}
    for layer in LAYERS:
        out["per_layer"][layer] = stats(np.array([l == layer for l in layers]))
    return out


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 on training rows (ties resolved to the lowest)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(scores):
        p = scores >= t
        tp = np.sum(y & p)
        fp = np.sum(~y & p)
        fn = np.sum(y & ~p)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def out_of_fold(fn, folds, y_by_target: dict, layers: list[str], n: int) -> dict:
    """fn(train_idx, test_idx, y) -> (test_scores, threshold). Out-of-fold metrics per target."""
    result = {}
    for name, y in y_by_target.items():
        pred = np.zeros(n, dtype=bool)
        thresholds = []
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            s_test, t = fn(train, test, y)
            thresholds.append(round(float(t), 4))
            pred[test] = s_test >= t
        result[name] = binary_metrics(y, pred, layers)
        result[name]["fold_thresholds"] = thresholds
    return result


# ---------- (a) cosine ----------

def cosine_method(cos: np.ndarray, folds, y_by_target, layers):
    def fn(train, test, y):
        return cos[test], best_threshold(cos[train], y[train])
    return out_of_fold(fn, folds, y_by_target, layers, len(cos))


# ---------- (c) features ----------

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def is_neg(tok: str) -> bool:
    return tok in NEG or tok.endswith("n't")


def features(pairs: list[dict], cos: np.ndarray) -> np.ndarray:
    rows = []
    for p, c in zip(pairs, cos):
        tr, tg = tokens(p["reference"]), tokens(p["generated"])
        sr, sg = set(tr), set(tg)
        jac = len(sr & sg) / len(sr | sg) if sr | sg else 0.0
        lr = min(len(tr), len(tg)) / max(len(tr), len(tg)) if tr and tg else 0.0
        neg_r = any(is_neg(t) for t in tr)
        neg_g = any(is_neg(t) for t in tg)
        onehot = [float(p["layer"] == l) for l in LAYERS]
        rows.append([float(c), *onehot, jac, lr, float(neg_r != neg_g)])
    return np.array(rows, dtype=float)


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 3000, lr: float = 0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = p - yf
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def predict_logreg(model, X):
    w, b = model
    return 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))


def fit_stumps(X: np.ndarray, y: np.ndarray, rounds: int = 150, lr: float = 0.1):
    """Logistic-loss gradient boosting with depth-1 trees (Newton leaf values)."""
    yf = y.astype(float)
    f0 = float(np.log(yf.mean() / (1 - yf.mean())))
    F = np.full(len(y), f0)
    stumps = []
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-F))
        g = yf - p
        h = p * (1 - p) + 1e-9
        best = None
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j])[:-1]:
                left = X[:, j] <= t
                gl, hl = g[left].sum(), h[left].sum()
                gr, hr = g[~left].sum(), h[~left].sum()
                gain = gl * gl / hl + gr * gr / hr
                if best is None or gain > best[0]:
                    best = (gain, j, float(t), gl / hl, gr / hr)
        if best is None:
            break
        _, j, t, vl, vr = best
        F += lr * np.where(X[:, j] <= t, vl, vr)
        stumps.append((j, t, vl, vr))
    return f0, stumps, lr


def predict_stumps(model, X):
    f0, stumps, lr = model
    F = np.full(len(X), f0)
    for j, t, vl, vr in stumps:
        F += lr * np.where(X[:, j] <= t, vl, vr)
    return 1 / (1 + np.exp(-F))


def feature_method(pairs, cos, folds, y_by_target, layers):
    X_all = features(pairs, cos)
    results = {}
    for kind in ("logreg", "gbm_stumps"):
        def fn(train, test, y, kind=kind):
            if kind == "logreg":
                mu = X_all[train].mean(0)
                sd = X_all[train].std(0) + 1e-9
                m = fit_logreg((X_all[train] - mu) / sd, y[train])
                pr = predict_logreg(m, (X_all[test] - mu) / sd)
            else:
                m = fit_stumps(X_all[train], y[train])
                pr = predict_stumps(m, X_all[test])
            return pr, 0.5
        results[kind] = out_of_fold(fn, folds, y_by_target, layers, len(pairs))
    return results


# ---------- (b) fine-tuned distilroberta ----------

def finetune_predict(model_dir: str, train: list[dict], test: list[dict], epochs: int,
                     lr: float, batch: int, seed: int) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=4, ignore_mismatched_sizes=True
    )

    def encode(rows):
        return tok(
            [f"{r['layer']}: {r['reference']}" for r in rows],
            [r["generated"] for r in rows],
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )

    y_train = torch.tensor([LABELS4.index(r["label"]) for r in train])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch - 1) // batch)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps)
    )
    order = list(range(len(train)))
    model.train()
    for _ in range(epochs):
        random.shuffle(order)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            enc = encode([train[j] for j in idx])
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(test), 32):
            out.append(torch.softmax(model(**encode(test[i : i + 32])).logits, dim=-1).numpy())
    return np.concatenate(out)


# ---------- (d) zero-shot NLI ----------

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def nli_entailment_probs(model_name: str, premises: list[str], hypotheses: list[str],
                         batch: int = 32) -> np.ndarray:
    """P(entailment | premise, hypothesis) from an off-the-shelf NLI cross-encoder.

    No parameters are trained on the calibration labels, so the only fitted
    quantity in CV is the decision threshold (chosen on train folds).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    ent_idx = [int(k) for k, v in model.config.id2label.items() if v.lower() == "entailment"]
    if len(ent_idx) != 1:
        raise SystemExit(f"no unique entailment label in {model_name}: {model.config.id2label}")
    out = []
    with torch.no_grad():
        for i in range(0, len(premises), batch):
            enc = tok(premises[i : i + batch], hypotheses[i : i + batch], padding=True,
                      truncation=True, max_length=256, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)
            out.append(probs[:, ent_idx[0]].numpy())
    return np.concatenate(out)


def nli_method(pairs, folds, y_by_target, layers, model_name: str) -> dict:
    refs = [p["reference"] for p in pairs]
    gens = [p["generated"] for p in pairs]
    # reference -> generated: reference is the premise, generated the hypothesis
    fwd = nli_entailment_probs(model_name, refs, gens)
    # generated -> reference
    rev = nli_entailment_probs(model_name, gens, refs)
    variants = {
        "nli_min_both_directions": np.minimum(fwd, rev),
        "nli_mean_both_directions": (fwd + rev) / 2.0,
    }
    out = {}
    for name, scores in variants.items():
        # same threshold-on-score CV as the cosine baseline (threshold fit on train folds)
        out[name] = out_of_fold(
            lambda train, test, y, s=scores: (s[test], best_threshold(s[train], y[train])),
            folds, y_by_target, layers, len(pairs),
        )
    return out


def run_nli(args, pairs, folds, y_by_target, layers, cos) -> None:
    t0 = time.time()
    methods = {"cosine_recomputed": cosine_method(cos, folds, y_by_target, layers)}
    methods.update(nli_method(pairs, folds, y_by_target, layers, args.nli_model))
    stored = json.loads((args.base / "matcher-v1.json").read_text())["methods"]
    results = {
        "task": "5-fold grouped-by-view_id zero-shot NLI matcher vs cosine (distill-v1)",
        "n_pairs": len(pairs),
        "n_views": len(set(p["view_id"] for p in pairs)),
        "label_counts": {l: sum(p["label"] == l for p in pairs) for l in LABELS4},
        "nli_model": args.nli_model,
        "nli_directions": {"fwd": "premise=reference, hypothesis=generated",
                           "rev": "premise=generated, hypothesis=reference"},
        "scores": "P(entailment); min or mean over both directions",
        "threshold": "maximises F1 on train folds; applied to held-out fold",
        "targets": {"strict": "label == same",
                    "multi": "label in {same, broader, narrower}"},
        "methods": methods,
        "stored_baseline_matcher_v1": {"a_minilm_cosine": stored["a_minilm_cosine"]},
        "device": "cpu",
        "runtime_seconds": None,
    }
    results["runtime_seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    summary = {m: {t: (r[t]["overall"]["f1"], r[t]["overall"]["accuracy"]) for t in r}
               for m, r in methods.items()}
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_B)
    ap.add_argument("--out", type=Path, default=DEFAULT_B / "matcher-v1.json")
    ap.add_argument("--encoder", default="distilbert/distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nli", action="store_true",
                    help="run only the zero-shot NLI matcher (no fine-tuning)")
    ap.add_argument("--nli-model", default=NLI_MODEL)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    pairs = load_pairs(args.base)
    view_ids = [p["view_id"] for p in pairs]
    layers = [p["layer"] for p in pairs]
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise SystemExit(f"unexpected layers {unknown}")
    folds = make_folds(view_ids, k=5, seed=args.seed)
    y_strict = np.array([p["label"] == "same" for p in pairs])
    y_multi = np.array([p["label"] in ("same", "broader", "narrower") for p in pairs])
    y_by_target = {"strict": y_strict, "multi": y_multi}
    cos = np.array([p["cosine"] for p in pairs], dtype=float)

    if args.nli:
        if args.out == DEFAULT_B / "matcher-v1.json":
            args.out = DEFAULT_B / "matcher-nli-v1.json"
        run_nli(args, pairs, folds, y_by_target, layers, cos)
        return

    model_dir = resolve_model(args.encoder)

    results: dict = {
        "task": "5-fold grouped-by-view_id matcher comparison (distill-v1)",
        "n_pairs": len(pairs),
        "n_views": len(set(view_ids)),
        "label_counts": {l: sum(p["label"] == l for p in pairs) for l in LABELS4},
        "fold_sizes": [int(len(f)) for f in folds],
        "cosine_model": "sentence-transformers/all-MiniLM-L6-v2 (precomputed)",
        "encoder": args.encoder,
        "finetune": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                     "max_length": 192, "head": "4-way same/broader/narrower/different",
                     "input": "layer: reference [SEP] generated",
                     "decision": "P(same)>=0.5 strict; 1-P(different)>=0.5 multi"},
        "feature_set": ["cosine", "layer_onehot", "token_jaccard", "length_ratio", "negation_mismatch"],
        "methods": {},
        "device": "cpu",
        "offline": True,
    }

    t0 = time.time()
    results["methods"]["a_minilm_cosine"] = cosine_method(cos, folds, y_by_target, layers)
    feat = feature_method(pairs, cos, folds, y_by_target, layers)
    results["methods"]["c_logreg"] = feat["logreg"]
    results["methods"]["c_gbm_stumps"] = feat["gbm_stumps"]
    print(f"classical done {time.time() - t0:.0f}s", flush=True)

    ft_strict = np.zeros(len(pairs))
    ft_multi = np.zeros(len(pairs))
    for fi, test in enumerate(folds):
        train_rows = [pairs[i] for i in np.setdiff1d(np.arange(len(pairs)), test)]
        test_rows = [pairs[i] for i in test]
        probs = finetune_predict(model_dir, train_rows, test_rows, args.epochs, args.lr,
                                 args.batch, args.seed + fi)
        ft_strict[test] = probs[:, 0]
        ft_multi[test] = 1.0 - probs[:, 3]
        print(f"fold {fi} fine-tuned {time.time() - t0:.0f}s", flush=True)
    ft = {
        "strict": binary_metrics(y_strict, ft_strict >= 0.5, layers),
        "multi": binary_metrics(y_multi, ft_multi >= 0.5, layers),
    }
    ft["strict"]["fold_thresholds"] = [0.5] * 5
    ft["multi"]["fold_thresholds"] = [0.5] * 5
    results["methods"]["b_distilroberta_finetuned"] = ft
    results["runtime_seconds"] = round(time.time() - t0, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({m: {k: (v["overall"]["f1"], v["overall"]["accuracy"]) for k, v in r.items()}
                      for m, r in results["methods"].items()}, indent=1))


if __name__ == "__main__":
    main()
