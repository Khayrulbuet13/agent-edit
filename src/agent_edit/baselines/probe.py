"""Linear probe (multinomial logistic regression, K classes + no object) on query embeddings."""

from __future__ import annotations

import torch

from .. import config
from .common import DEV, NO_OBJECT_WEIGHT


def _probe(X, Y, K, wd, device="cpu"):
    X, Y = X.to(device), Y.to(device)
    W = torch.zeros(K + 1, X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(K + 1, device=device, requires_grad=True)
    cw = torch.ones(K + 1, device=device)
    cw[K] = NO_OBJECT_WEIGHT
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W.T + b, Y, weight=cw) + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"class_rows.weight": W.detach().cpu(), "class_rows.bias": b.detach().cpu()}


def fit_c2(ims, wd, ctx):
    X, Y = ims.xy(True)
    return _probe(X, Y, ims.K, wd, DEV), {}


if config.has_module("baselines.probe_lab"):  # names of an optional sibling module, when the tree has it
    from .probe_lab import *  # noqa: F403
