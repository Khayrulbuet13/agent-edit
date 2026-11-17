"""Cosine-head baseline (`c5r`): Adam on the PMT class rows + no-object row, cross-entropy over K+1 per query.

A TFA-style cosine head, logit = s cos(q, w) + b, imprinted from the class means.
"""

from __future__ import annotations

import torch

from .. import config
from .common import DEV, NO_OBJECT_WEIGHT, fast_counts


def _cos_traj(ims, scale, lr, steps_wanted):
    """Full-batch Adam on the cosine head; {steps: weights} at each requested step count."""
    X, Y = ims.xy(True)
    K = ims.K
    W0 = torch.zeros(K + 1, X.shape[1])
    for k in range(K + 1):
        sel = X[Y == k]
        W0[k] = sel.mean(0) if len(sel) else X.mean(0)
    X, Y = X.to(DEV), Y.to(DEV)
    Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-6)
    W = W0.to(DEV).requires_grad_(True)
    b = torch.zeros(K + 1, device=DEV, requires_grad=True)
    cw = torch.ones(K + 1, device=DEV)
    cw[K] = NO_OBJECT_WEIGHT
    opt = torch.optim.Adam([W, b], lr=lr)
    out = {}
    for s in range(1, max(steps_wanted) + 1):
        opt.zero_grad()
        Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-6)
        torch.nn.functional.cross_entropy(scale * Xn @ Wn.T + b, Y, weight=cw).backward()
        opt.step()
        if s in steps_wanted:
            # zero linear rows: the whole head lives in the cos_* terms (backend.extra_logit_fn)
            out[s] = {"class_rows.weight": torch.zeros(K + 1, X.shape[1]), "class_rows.bias": b.detach().cpu().clone(),
                      "cos_rows": W.detach().cpu().clone(), "cos_scale": torch.tensor(scale)}
    return out


def fit_c5r(ims, cfg, ctx):
    sc, lr, st = cfg
    return _cos_traj(ims, sc, lr, [st])[st], {}


def c5r_fold_counts(fi, hi, grid, ctx):
    """One Adam trajectory per (scale, lr) gives every step count."""
    out = [None] * len(grid)
    for sc, lr in sorted({g[:2] for g in grid}):
        sts = [g[2] for g in grid if g[:2] == (sc, lr)]
        tr = _cos_traj(fi, sc, lr, sts)
        for st in sts:
            out[grid.index((sc, lr, st))] = fast_counts(hi, tr[st])
    return out


if config.has_module("baselines.adam_lab"):  # names of an optional sibling module, when the tree has it
    from .adam_lab import *  # noqa: F403
