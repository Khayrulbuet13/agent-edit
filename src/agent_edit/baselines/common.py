"""Shared pieces of the no-agent PMT class-head baselines. Only train images are used; eval goes
through backends.pmt.evaluate.

A method's grid point is picked by cv: one fixed grid per method (GRIDS), stratified 5-fold CV on the
train images, refit on all of them. Scored with `fast_counts`: an exact copy of the official scorer on
sets with one true instance per image (the fine-grained sets); on sets with several instances per image
(xview-ships-10), a per-image per-class greedy match over the same kept predictions -- a fit-time proxy
for picking hyperparameters, not the reported number (that always comes from backends.pmt.evaluate's
real matcher).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

from .. import config, data
from ..backends.pmt import backend as PB
from ..backends.pmt import cache as C
from ..backends.pmt import head as H
from ..metrics.f1 import f1_from_counts
from ..metrics.match import gt_by_class

NO_OBJECT_WEIGHT = 0.1  # PMT's own no_object_coefficient


def out_dir() -> Path:
    return config.results_dir() / "pmt_fg/baselines"


FOLDS = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
THR = PB.SCORE_THRESHOLD
GATE_GRID = np.arange(-30.0, 40.01, 0.5)  # no-object bias offsets; class means need large ones
GRIDS = {
    "C2": [0.0] + [float(w) for w in np.logspace(-8, -2, 15)],
    # scale 20 is TFA's; PMT queries share one big direction, so cosines are compressed and need more
    "C5R": [(sc, lr, st) for sc in (20.0, 50.0, 100.0, 200.0) for lr in (1e-2, 3e-2) for st in (500, 2000)],
}


class Images:
    """Train images of a PMT set. Per image: q [200, D], mscore [200]. Per true instance (one or
    more per image -- xview-ships-10's chips can hold several vessels): inst_img (which image),
    inst_gt (class index), inst_iou [200] (IoU of every query in that image against this instance).

    `single` is True when every image holds exactly one instance (every fine-grained set).
    Row-indexed callers (recipe.py, the formula baseline) read inst_img/inst_gt/inst_iou instead of
    a bare gt/iou, so they run unchanged on single or multi-instance sets: on a single-instance set
    inst_img is arange(n), so e.g. `ims.q[ims.inst_img]` is just `ims.q`.
    """

    def __init__(self, project, recs):
        root = C.source_root(project, "pmt")
        self._fill(project, recs, [C.load(root, r["image_key"]) for r in recs])

    @classmethod
    def from_entries(cls, project, recs, entries):
        o = object.__new__(cls)
        o._fill(project, recs, entries)
        return o

    def _fill(self, project, recs, entries):
        cids = [c["class_id"] for c in data.class_table(project)]
        box = project in PB.BOX_PROJECTS
        Q, M, inst_img, inst_gt, inst_iou = [], [], [], [], []
        single = True
        for i, (rec, e) in enumerate(zip(recs, entries)):
            gtc = gt_by_class(rec)
            g = [(j, r) for j, c in enumerate(cids) for r in gtc.get(c, [])]
            if not g:
                raise ValueError(f"{project} {rec['image_key']} has 0 true instances; the cv protocol needs "
                                 "at least one per image")
            if len(g) > 1:
                single = False
            if e.get("top_k", H.TOP_K) != H.TOP_K or "tiles" in e:
                raise ValueError("the cv protocol needs whole-image PMT entries")
            dt = [PB.box_rle(r) for r in e["rles"]] if box else e["rles"]
            gt = [PB.box_rle(r) for _, r in g] if box else [r for _, r in g]
            iou_mat = np.asarray(mask_utils.iou(dt, gt, [0] * len(gt))).reshape(len(dt), len(gt))  # [200, len(g)]
            Q.append(e["q"])
            M.append(e["mscore"])
            for j, (cid_idx, _) in enumerate(g):
                inst_img.append(i)
                inst_gt.append(cid_idx)
                inst_iou.append(torch.from_numpy(iou_mat[:, j]).float())
        self.q, self.mscore = torch.stack(Q), torch.stack(M)
        self.inst_img = torch.tensor(inst_img, dtype=torch.long)
        self.inst_gt = torch.tensor(inst_gt, dtype=torch.long)
        self.inst_iou = torch.stack(inst_iou)
        self.single = single
        self.keys, self.K, self.project = [r["image_key"] for r in recs], len(cids), project
        self._d = None

    def sub(self, idx):
        idx = list(idx)
        o = object.__new__(Images)
        o.q, o.mscore = self.q[idx], self.mscore[idx]
        pos = {img_i: new_i for new_i, img_i in enumerate(idx)}
        keep = [m for m, img_i in enumerate(self.inst_img.tolist()) if img_i in pos]
        o.inst_img = torch.tensor([pos[int(self.inst_img[m])] for m in keep], dtype=torch.long)
        o.inst_gt, o.inst_iou = self.inst_gt[keep], self.inst_iou[keep]
        o.single = self.single
        o.keys, o.K, o.project = [self.keys[i] for i in idx], self.K, self.project
        o._d = None
        return o

    def __len__(self):
        return len(self.keys)

    def xy(self, one_to_one=True):
        """Flat queries and targets (K = no object), one row of Y per query of every image.
        one_to_one: each image's instances claim distinct queries (Hungarian on IoU -- exact
        argmax when an image has one instance); else every query at IoU >= 0.5 to its
        best-overlapping instance."""
        n, nq = self.q.shape[:2]
        Y = torch.full((n, nq), self.K, dtype=torch.long)
        if self.single:
            iou, gt = self.inst_iou, self.inst_gt
            if one_to_one:
                best = iou.argmax(1)
                ok = iou[torch.arange(n), best] >= 0.5
                Y[torch.arange(n)[ok], best[ok]] = gt[ok]
            else:
                hit = iou >= 0.5
                Y[hit] = gt[:, None].expand(n, nq)[hit]
            return self.q.reshape(n * nq, -1), Y.reshape(-1)
        for i in range(n):
            rows = torch.where(self.inst_img == i)[0]
            if not len(rows):
                continue
            iou_i, gt_i = self.inst_iou[rows], self.inst_gt[rows]
            if one_to_one:
                from scipy.optimize import linear_sum_assignment

                gi, qi = linear_sum_assignment((-iou_i).numpy())
                # ok must be a numpy bool array here: numpy reads a torch bool tensor as an int
                # index (True -> 1), not a mask, and qi/gi are numpy (from scipy).
                ok = (iou_i[gi, qi] >= 0.5).numpy()
                Y[i, qi[ok]] = gt_i[gi[ok]]
            else:
                hit = iou_i >= 0.5
                if hit.any():
                    best_row = torch.where(hit, iou_i, torch.full_like(iou_i, -1.0)).argmax(0)
                    cols = hit.any(0)
                    Y[i, cols] = gt_i[best_row[cols]]
        return self.q.reshape(n * nq, -1), Y.reshape(-1)

    def dev(self):
        if self._d is None:
            self._d = (self.q.to(DEV), self.mscore.to(DEV))
        return self._d


def folds(n_img, K):
    """Train lists interleave classes (image i is class i % K at round i // K), so fold j holds out
    within-class positions j, j + FOLDS, ..."""
    pos = np.arange(n_img) // K
    return [np.where(pos % FOLDS == j)[0] for j in range(FOLDS)]


@torch.no_grad()
def logits_of(ims: Images, w: dict) -> torch.Tensor:
    """[n, 200, K+1] head logits on DEV, as predict() computes them."""
    q = ims.dev()[0]
    n, nq, D = q.shape
    W, b = w["class_rows.weight"].float().to(DEV), w["class_rows.bias"].float().to(DEV)
    fn = PB.extra_logit_fn({k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in w.items()})
    out = []
    for s in range(0, n, 64):
        x = q[s:s + 64].reshape(-1, D)
        lg = x @ W.T + b
        if fn is not None:
            lg = lg + fn(x)
        out.append(lg.reshape(-1, nq, W.shape[0]))
    return torch.cat(out)


@torch.no_grad()
def counts_from_logits(ims: Images, lg: torch.Tensor) -> np.ndarray:
    """(K, 3) [tp, fp, n_gt], the numbers agent_edit.metrics.f1.counts gives on these images.

    ims.single (one true instance per image, every fine-grained set): greedy matching
    reduces to "is any kept prediction of the true class at IoU >= 0.5" -- the original fast
    formula, fully vectorized.

    Multi-instance images (xview-ships-10's chips can hold several vessels; sets of
    small cells can hold over a thousand in one image): same kept predictions, credited to true instances with a
    per-image per-class one-to-one match (Hungarian on IoU, maximizing total IoU -- a vectorized
    stand-in for metrics.match.greedy_match's score-order walk, same idea as Images.xy's; exact
    when a class has one instance in that image) so two instances of one class in one image can't
    both be paid by the same single detection. This is a fit-time proxy for picking
    hyperparameters, not the reported number -- that always comes from backends.pmt.evaluate's
    real matcher on the actual masks."""
    ms = ims.dev()[1]
    K = ims.K
    n, nq, _ = lg.shape
    p = lg.softmax(-1)[..., :K].reshape(n, nq * K)
    top, idx = p.topk(H.TOP_K, dim=1)
    lab, qi = idx % K, idx // K
    sc = top * ms.gather(1, qi)
    keep = sc >= THR
    npred = torch.bincount(lab[keep], minlength=K)
    inst_gt = ims.inst_gt.to(lab.device)
    ngt = torch.bincount(inst_gt, minlength=K)

    if ims.single:
        iou = ims.inst_iou.to(lab.device)
        good = keep & (lab == inst_gt[:, None]) & (iou.gather(1, qi) >= 0.5)
        tp = torch.bincount(inst_gt[good.any(1)], minlength=K)
        return torch.stack([tp, npred - tp, ngt], 1).cpu().numpy()

    from scipy.optimize import linear_sum_assignment

    lab_cpu, qi_cpu, keep_cpu = lab.cpu(), qi.cpu(), keep.cpu()
    by_img: dict[int, list[int]] = {}
    for m, im in enumerate(ims.inst_img.tolist()):
        by_img.setdefault(im, []).append(m)
    tp = torch.zeros(K, dtype=torch.long)
    for im, ms_i in by_img.items():
        lab_i, qi_i, keep_i = lab_cpu[im], qi_cpu[im], keep_cpu[im]
        for c in sorted({int(ims.inst_gt[m]) for m in ms_i}):
            rows = [m for m in ms_i if int(ims.inst_gt[m]) == c]
            cand = torch.where(keep_i & (lab_i == c))[0]
            if not len(cand):
                continue
            block = ims.inst_iou[rows][:, qi_i[cand]]  # [len(rows), len(cand)]
            ri, ci = linear_sum_assignment(-block.numpy())
            tp[c] += int((block[ri, ci] >= 0.5).sum())
    return torch.stack([tp, npred.cpu() - tp, ngt.cpu()], 1).numpy()


def fast_counts(ims, w) -> np.ndarray:
    return counts_from_logits(ims, logits_of(ims, w))


def macro_f1(Cn) -> float:
    return f1_from_counts(np.asarray(Cn))[0]


def fast_f1(ims, w) -> float:
    return macro_f1(fast_counts(ims, w))


@lru_cache(maxsize=1)
def _stock():
    return H.load_head_weights()


def stock():
    return {k: v.clone() for k, v in _stock().items() if k.startswith("class_head")}


def start(project):
    """PB.start_weights(project, 'pmt') without reloading the checkpoint."""
    st = _stock()
    rows = PB.slot_rows(project) + [H.NO_OBJECT]
    return {"class_rows.weight": st["class_head.weight"][rows].clone(),
            "class_rows.bias": st["class_head.bias"][rows].clone()}


def gate_sweep(ims, w, unit=1.0, grid=GATE_GRID, rows="noobj"):
    """Best fit-image F1 over shifts of the no-object bias (rows='noobj') or of every class bias
    downwards (rows='classes'). Ties go to the smallest shift."""
    lg0 = logits_of(ims, w)
    best = None
    for d in grid:
        lg = lg0.clone()
        if rows == "noobj":
            lg[..., -1] += unit * d
        else:
            lg[..., :-1] -= unit * d
        f = macro_f1(counts_from_logits(ims, lg))
        if best is None or f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(d) < abs(best[1])):
            best = (f, float(d))
    ww = {k: v.clone() if torch.is_tensor(v) else v for k, v in w.items()}
    if rows == "noobj":
        ww["class_rows.bias"][-1] += unit * best[1]
    else:
        ww["class_rows.bias"][:-1] -= unit * best[1]
    return ww, {"gate_shift": best[1] * unit, "fit_f1": best[0]}


def cv_pick(grid, fit, ims, ctx, fold_counts=None):
    """Grid point with the best macro-F1 over the pooled held-out counts of the 5 folds; ties go to
    the earlier point. fold_counts(fit_ims, held_ims, grid, ctx) -> one count table per grid point,
    for methods that share work across grid points; otherwise each point is fit on its own."""
    tot = [np.zeros((ims.K, 3), np.int64) for _ in grid]
    for ho in folds(len(ims), ims.K):
        if not len(ho):
            continue
        fi = ims.sub(np.setdiff1d(np.arange(len(ims)), ho).tolist())
        hi = ims.sub(ho.tolist())
        cs = fold_counts(fi, hi, grid, ctx) if fold_counts else [fast_counts(hi, fit(fi, g, ctx)[0]) for g in grid]
        for t, c in zip(tot, cs):
            t += c
    scores = [macro_f1(t) for t in tot]
    best = int(np.argmax(scores))
    return grid[best], {str(g): round(s, 6) for g, s in zip(grid, scores)}


if config.has_module("baselines.common_lab"):  # names of an optional sibling module, when the tree has it
    from .common_lab import *  # noqa: F403
