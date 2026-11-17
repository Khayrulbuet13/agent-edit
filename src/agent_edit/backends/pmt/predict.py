"""Class-head predictions from a cached PMT entry, on CPU, for any class-row weights.

`predict` reproduces head.postprocess from the cache; masks come back as RLE.
"""

from __future__ import annotations

import torch

from . import head as H
from .tiling import dedupe_keep, paste


def class_logits(entry: dict, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return entry["q"] @ w.T + b


def predict(entry: dict, w: torch.Tensor, b: torch.Tensor, slots=None, threshold: float = 0.5,
            top_k: int = H.TOP_K, logit_fn=None) -> dict:
    """w, b: class head weight [R, EMBED] and bias [R]; slots index rows of w, and the last
    row of w is the no-object row."""
    rows = list(range(w.shape[0] - 1)) if slots is None else list(slots)
    top_k = entry.get("top_k", top_k)
    logits = class_logits(entry, w[rows + [w.shape[0] - 1]], b[rows + [w.shape[0] - 1]])
    if logit_fn is not None:  # baselines whose classifier is not a linear head (Tip-Adapter, cosine head)
        logits = logits + logit_fn(entry["q"])
    scores = logits.softmax(-1)[:, :-1]
    k = min(top_k, scores.numel())
    top, idx = scores.flatten().topk(k, sorted=False)
    labels, q = idx % len(rows), idx // len(rows)
    s = top * entry["mscore"][q]
    keep = s >= threshold
    q = q[keep]
    return {"rles": [entry["rles"][i] for i in q.tolist()], "labels": labels[keep], "scores": s[keep], "queries": q}


def predict_tiled(entry: dict, w, b, threshold: float = 0.5, top_k: int = H.TOP_K, dedupe: float = 0.5,
                  logit_fn=None) -> dict:
    """Per-tile predict, then drop a prediction when a better-scored one of the same class covers it:
    IoU >= dedupe, or >= 80% of the smaller mask (a cell cut by a tile border)."""
    parts = [predict(t, w, b, threshold=threshold, top_k=top_k, logit_fn=logit_fn) for t in entry["tiles"]]
    for t, p in zip(entry["tiles"], parts):
        if t.get("local"):
            p["rles"] = [paste(r, t["window"], entry["hw"]) for r in p["rles"]]
    rles = [r for p in parts for r in p["rles"]]
    if not rles:
        return {"rles": [], "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0),
                "queries": torch.zeros(0, dtype=torch.long)}
    labels = torch.cat([p["labels"] for p in parts])
    scores = torch.cat([p["scores"] for p in parts])
    queries = torch.cat([p["queries"] + j * H.NUM_Q for j, p in enumerate(parts)])
    order = scores.argsort(descending=True).tolist()
    keep = dedupe_keep(rles, labels.tolist(), order, dedupe)
    k = torch.tensor(keep, dtype=torch.long)
    return {"rles": [rles[i] for i in keep], "labels": labels[k], "scores": scores[k], "queries": queries[k]}
