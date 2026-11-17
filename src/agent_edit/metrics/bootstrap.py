"""Paired bootstrap over eval images for the pooled macro-F1."""

from __future__ import annotations

import numpy as np

from .f1 import f1_from_counts

BOOT = 5000


def bootstrap_diff(Ca, Cb, n=BOOT, seed=0) -> dict:
    """macro-F1(b) - macro-F1(a), resampling images (the pooled metric, not a mean of per-image F1)."""
    rng = np.random.RandomState(seed)
    N = Ca.shape[0]
    d = np.empty(n)
    for k in range(n):
        ix = rng.randint(0, N, N)
        d[k] = f1_from_counts(Cb[ix])[0] - f1_from_counts(Ca[ix])[0]
    pt = f1_from_counts(Cb)[0] - f1_from_counts(Ca)[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": pt, "ci95": [float(lo), float(hi)], "p_le_0": float((d <= 0).mean()), "n_boot": n}


def f1_ci(Cn, n=BOOT, seed=0) -> list[float]:
    """95% interval of macro-F1 over images resampled with replacement."""
    rng = np.random.RandomState(seed)
    N = Cn.shape[0]
    v = [f1_from_counts(Cn[rng.randint(0, N, N)])[0] for _ in range(n)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]
