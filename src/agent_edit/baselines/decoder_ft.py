"""Full decoder fine-tuning (`c5f`; result files are tagged `C5F`): AdamW on the whole PMT decoder (GPU),
from the stock decoder + the linear-probe rows (`c2`), i.e. LP-FT.

Same cv protocol as the head baselines: a grid (lr x epochs), 5 folds of the train images (one
fold per job if you like), refit on all train images with the picked point. The eval images are
then re-encoded with the trained decoder and scored by the official scorer.

The grid is lr in 1e-5..1e-3 (log-spaced) x epochs in 1..24 (GRID below). `order` is the batch
shuffle seed; it is independent of `seed` (the data draw), so the same train split can be run
under several batch orders to see how much the result swings.

Loss: PMT's mask-classification loss on the last block only (no deep supervision). Per image, every
true instance (xview-ships-10's chips can hold several) claims its own query by a Hungarian match
on (-2 p_class + 5 BCE + 5 dice) over the 200 queries -- argmin when an image has one instance; CE
over the K slot rows + no-object (no-object weight 0.1) on every query, BCE + dice summed over the
matched (query, instance) pairs (full 320x320 logits, no point sampling). Box sets train masks
toward the box rectangle.

    agent-edit baselines c5f PROJECT --seed S --n N --fold J   -> <out>/<project>/C5F_s<S>_N<N>_fold<J>.json
    agent-edit baselines c5f PROJECT --seed S --n N            -> missing folds, then C5F_s<S>_N<N>.json
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from .. import data
from ..backends.pmt import backend as PB
from ..backends.pmt import cache as C
from ..backends.pmt import head as H
from ..metrics.match import gt_by_class
from .common import FOLDS, NO_OBJECT_WEIGHT, Images, fast_counts, folds, macro_f1, out_dir, start
from .probe import fit_c2

LRS = (1e-5, 2.2e-5, 4.6e-5, 1e-4, 2.2e-4, 4.6e-4, 1e-3)
EPOCHS = (1, 2, 4, 8, 16, 24)
GRID = [(lr, ep) for lr in LRS for ep in EPOCHS]
BATCH = 2
WD = 0.05
DENSE_MATCH_MAX = 128  # true instances per image up to which the match cost is computed elementwise

# LwF: distill the stock decoder's own outputs on the same train images, weighed by kd_weight.
# Reuses the (lr, epochs) that full fine-tuning already picked for the cell (that search is expensive; redoing it
# for every kd_weight is not) and grids only the new knob.
# kd_loss's natural scale is much bigger than the task loss: weights of 1 and above wipe out the
# task, so the grid starts far below that.
LWF_LAMBDAS = (0.01, 0.05, 0.2)


def _out(root) -> Path:
    return Path(root or out_dir())


def tag(seed, n, order=0) -> str:
    """order 0 gives the plain C5F_s<S>_N<N> name; other orders add _o<order>."""
    return f"C5F_s{seed}_N{n}" if order == 0 else f"C5F_s{seed}_o{order}_N{n}"


def pick(cv_f1: dict) -> tuple:
    """Grid point with the best cv macro-F1; ties go to the earlier point (GRID order)."""
    return max(GRID, key=lambda g: (cv_f1[str(g)], -GRID.index(g)))


def gt_info(project, rec):
    """[(class slot, rle), ...], one or more true instances (xview-ships-10's chips can hold
    several vessels)."""
    cids = [c["class_id"] for c in data.class_table(project)]
    g = [(j, r) for j, c in enumerate(cids) for r in gt_by_class(rec).get(c, [])]
    if not g:
        raise ValueError(f"{project} {rec['image_key']} has 0 true instances")
    out = []
    for j, r in g:
        if project in PB.BOX_PROJECTS:
            r = PB.box_rle(r)
            r = {"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}
        out.append((j, r))
    return out


def gt_mask_320(rle, hw):
    """True mask in the decoder's 320x320 mask-logit frame (image at top-left, scaled, padded)."""
    m = torch.from_numpy(mask_utils.decode(rle)).float()[None, None]
    sh, sw = H.Backbone.scaled_size(hw)
    m = F.interpolate(m, (max(1, round(sh / 4)), max(1, round(sw / 4))), mode="bilinear")[0, 0]
    out = torch.zeros(4 * H.GRID, 4 * H.GRID)
    out[: m.shape[0], : m.shape[1]] = (m >= 0.5).float()
    return out


class Feats:
    def __init__(self, project, recs, bb, gt=True):
        self.recs = recs
        # gt/gm are python lists (one true-instance count per image can differ): gt[i] is [Gi]
        # classes, gm[i] is [Gi, h, w] true masks, both on "cuda".
        f, self.hw, self.gt, self.gm = [], [], [], []
        for rec in recs:
            img = np.asarray(data.open_rgb(rec["path"]))
            x, rope = bb.features(img)
            f.append(x.half())
            self.hw.append(tuple(img.shape[:2]))
            if not gt:  # eval images are only re-encoded; their true masks would fill the GPU on sets with many instances
                continue
            insts = gt_info(project, rec)
            self.gt.append(torch.tensor([j for j, _ in insts], dtype=torch.long, device="cuda"))
            self.gm.append(torch.stack([gt_mask_320(r, img.shape[:2]) for _, r in insts]).cuda())
        self.f = torch.stack(f)
        self.rope = rope


@lru_cache(maxsize=1)
def _dec():
    # functional_call never mutates it, and each build reads the whole checkpoint from disk
    return H.build_decoder("cuda")


@lru_cache(maxsize=1)
def _stock_weights():
    return {k: v.cuda() for k, v in H.load_head_weights("cpu").items()}


def fwd(dec, p, feats, rope):
    """H.head_forward in bf16 autocast: fp32 attention with PMT's (all-pass) mask is ~11 GB per call."""
    from torch.func import functional_call

    outs = [feats[:, i].transpose(1, 2).contiguous().transpose(1, 2) for i in range(feats.shape[1])]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        masks, classes = functional_call(dec, p, (outs, rope), strict=False)
    return masks[-1].float(), classes[-1].float()


def pair_costs(ml, gm):
    """BCE and dice cost of every (query, true instance) pair: ml [nq, h, w] mask logits,
    gm [G, h, w] true masks -> two [nq, G] tensors."""
    if gm.shape[0] <= DENSE_MATCH_MAX:
        # elementwise: the direct form, used while the instance count is small
        nq, G = ml.shape[0], gm.shape[0]
        sg = ml.sigmoid()
        ml_b, gm_b = ml[:, None].expand(-1, G, -1, -1), gm[None].expand(nq, -1, -1, -1)
        bce = F.binary_cross_entropy_with_logits(ml_b, gm_b, reduction="none").mean((2, 3))
        dice = 1 - (2 * (sg[:, None] * gm[None]).sum((2, 3)) + 1) / (
            sg.sum((1, 2))[:, None] + gm.sum((1, 2))[None] + 1)
        return bce, dice
    # matmul form: a [nq, G, h, w] tensor is 20 GB at G = 256 (sets of small cells can have over a thousand)
    x, g = ml.flatten(1), gm.flatten(1)
    neg = F.softplus(x)
    bce = (F.softplus(-x) @ g.T + neg.sum(1, keepdim=True) - neg @ g.T) / x.shape[1]
    sg = x.sigmoid()
    dice = 1 - (2 * sg @ g.T + 1) / (sg.sum(1)[:, None] + g.sum(1)[None] + 1)
    return bce, dice


def kd_loss(masks, cls, stock_masks, stock_cls) -> torch.Tensor:
    """LwF: keep the current decoder's output near the frozen stock decoder's, on the same images.
    KL on the full 81-row softmax (what COCO retention scores) + MSE on the mask logits."""
    kl = F.kl_div(cls.log_softmax(-1), stock_cls.softmax(-1), reduction="batchmean")
    return kl + F.mse_loss(masks, stock_masks)


def train(project, feats: Feats, idx, base, c2head, lr, epochs_wanted, seed=0, log=print, kd_weight=0.0):
    """{epoch: decoder weights} at each wanted epoch count of one AdamW run.
    kd_weight > 0: add the LwF distillation loss against the frozen stock decoder (same images)."""
    K = len(data.class_table(project))
    rows = PB.slot_rows(project) + [H.NO_OBJECT]
    dec = _dec()
    stock = _stock_weights() if kd_weight else None
    p = {k: v.clone() for k, v in base.items()}
    cw_ = p["class_head.weight"].clone()
    cb_ = p["class_head.bias"].clone()
    cw_[rows] = c2head["class_rows.weight"].cuda()
    cb_[rows] = c2head["class_rows.bias"].cuda()
    p["class_head.weight"], p["class_head.bias"] = cw_, cb_
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=lr, weight_decay=WD)
    cw = torch.ones(K + 1, device="cuda")
    cw[K] = NO_OBJECT_WEIGHT
    rows_t = torch.tensor(rows, device="cuda")
    g = torch.Generator().manual_seed(seed)
    out = {}
    idx = list(idx)
    t0 = time.time()
    for ep in range(1, max(epochs_wanted) + 1):
        perm = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            x = feats.f[b].float()
            masks, cls = fwd(dec, p, x, feats.rope)
            lg = cls[:, :, rows_t]
            loss = 0.0
            for i in range(len(b)):
                ml = masks[i].float()                       # [200, h, w]
                gt_i, gm_i = feats.gt[b[i]], feats.gm[b[i]]  # [Gi], [Gi, h, w] -- Gi >= 1
                nq = ml.shape[0]
                # Assignment first, gradient-free; loss terms below only for the matched pairs.
                with torch.no_grad():
                    bce_c, dice_c = pair_costs(ml.detach(), gm_i)
                    cost = -2 * lg[i].detach().softmax(-1)[:, gt_i] + 5 * bce_c + 5 * dice_c
                    qi, gi = linear_sum_assignment(cost.cpu().numpy())  # qi: query idx, gi: which of the Gi instances
                ml_m, gm_m = ml[qi], gm_i[gi]
                sg_m = ml_m.sigmoid()
                bce_m = F.binary_cross_entropy_with_logits(ml_m, gm_m, reduction="none").mean((1, 2))
                dice_m = 1 - (2 * (sg_m * gm_m).sum((1, 2)) + 1) / (sg_m.sum((1, 2)) + gm_m.sum((1, 2)) + 1)
                y = torch.full((nq,), K, device="cuda", dtype=torch.long)
                y[qi] = gt_i[gi]
                loss = loss + F.cross_entropy(lg[i], y, weight=cw) * 2 + 5 * bce_m.sum() + 5 * dice_m.sum()
            if kd_weight:
                with torch.no_grad():
                    stock_masks, stock_cls = fwd(dec, stock, x, feats.rope)
                loss = loss + kd_weight * len(b) * kd_loss(masks, cls, stock_masks, stock_cls)
            opt.zero_grad()
            (loss / len(b)).backward()
            opt.step()
        if ep in epochs_wanted:
            out[ep] = {k: v.detach().clone() for k, v in p.items()}
            log(f"  lr {lr} epoch {ep} loss {float(loss) / len(b):.3f} ({time.time() - t0:.0f}s)")
    return out


def head_of(project, p):
    rows = PB.slot_rows(project) + [H.NO_OBJECT]
    return {"class_rows.weight": p["class_head.weight"][rows].float().cpu(),
            "class_rows.bias": p["class_head.bias"][rows].float().cpu()}


@torch.no_grad()
def encode_all(feats: Feats, idx, p):
    dec = _dec()
    w = {k: v.float() for k, v in p.items()}
    return [C.encode(dec, w, feats.f[i].float(), feats.rope, feats.hw[i]) for i in idx]


def c2_head_for(project, seed, n, fit_ims, root=None):
    """Linear probe refit on the fit images with the weight decay the probe baseline (`c2`) picked for this cell."""
    jp = _out(root) / project / f"C2_s{seed}_N{n}.json"
    if not jp.exists():
        from .run import run_cell

        run_cell(project, seed, n, ["C2"], root)
    return fit_c2(fit_ims, float(json.loads(jp.read_text())["picked"]), {})[0]


def _features(project, rec_lists, gt=None):
    bb = H.Backbone("cuda")
    out = [Feats(project, recs, bb, g) for recs, g in zip(rec_lists, gt or [True] * len(rec_lists))]
    del bb
    torch.cuda.empty_cache()
    return out


def run_fold(project, seed, n, j, root=None, log=None, order=0) -> Path:
    log = log or (lambda s: print(s, flush=True))
    recs = data.load_split(project, "train", n=n, seed=seed)
    fl = folds(len(recs), len(data.class_table(project)))
    ho = fl[j].tolist()
    fit = np.setdiff1d(np.arange(len(recs)), fl[j]).tolist()
    feats, = _features(project, [recs])
    c2h = c2_head_for(project, seed, n, Images(project, recs).sub(fit), root)
    base = {k: v.cuda() for k, v in H.load_head_weights("cpu").items()}
    res = {}
    for lr in LRS:
        traj = train(project, feats, fit, base, c2h, lr, EPOCHS, seed=order, log=log)
        for ep, p in traj.items():
            ims = Images.from_entries(project, [recs[i] for i in ho], encode_all(feats, ho, p))
            res[str((lr, ep))] = fast_counts(ims, head_of(project, p)).tolist()
        del traj
        torch.cuda.empty_cache()
    d = _out(root) / project
    d.mkdir(parents=True, exist_ok=True)
    jp = d / f"{tag(seed, n, order)}_fold{j}.json"
    jp.write_text(json.dumps(res))
    return jp


def run_final(project, seed, n, root=None, cache_root=None, log=None, order=0) -> Path:
    """Pick from the fold files, refit on all train images, re-encode eval, score.
    cache_root: where the re-encoded eval images go (default next to the result)."""
    from ..backends.pmt import evaluate as E

    log = log or (lambda s: print(s, flush=True))
    d = _out(root) / project
    t = tag(seed, n, order)
    tot = {str(g): np.zeros((len(data.class_table(project)), 3), np.int64) for g in GRID}
    for j in range(FOLDS):
        fj = json.loads((d / f"{t}_fold{j}.json").read_text())
        for k in tot:
            tot[k] += np.asarray(fj[k])
    cv = {k: macro_f1(v) for k, v in tot.items()}
    best = pick(cv)
    recs = data.load_split(project, "train", n=n, seed=seed)
    ev_recs = data.load_split(project, "eval")
    feats, ev_feats = _features(project, [recs, ev_recs], gt=[True, False])
    c2h = c2_head_for(project, seed, n, Images(project, recs), root)
    base = {k: v.cuda() for k, v in H.load_head_weights("cpu").items()}
    t0 = time.time()
    p = train(project, feats, range(len(recs)), base, c2h, best[0], [best[1]], seed=order, log=log)[best[1]]
    tfit = time.time() - t0
    src = f"{t.lower()}"
    enc = Path(cache_root) if cache_root else d / f"{t}_evalcache"
    for rec, e in zip(ev_recs, encode_all(ev_feats, range(len(ev_recs)), p)):
        C.save(enc, rec["image_key"], e)
    ev = E.evaluate(project, head_of(project, p), ref=start(project), source=src, root=enc)
    torch.save({k: v.cpu() for k, v in p.items()}, d / f"{t}.decoder.pt")
    rec = {"method": "C5F", "project": project, "seed": seed, "n": n, "order": order, "picked": list(best),
           "cv_f1": {k: round(v, 6) for k, v in cv.items()}, "seconds_fit": round(tfit, 1),
           "note": "ref_macro_f1 / bootstrap_minus_ref use the stock rows on the re-encoded eval cache",
           "eval": {k: v for k, v in ev.items() if k != "counts"}, "eval_counts": ev["counts"]}
    jp = d / f"{t}.json"
    jp.write_text(json.dumps(rec, indent=1))
    log(f"C5F {project} s{seed} o{order} N{n}: picked {best} cv {cv[str(best)]:.3f} | eval F1 {ev['macro_f1']:.4f} "
        f"{[round(x, 3) for x in ev['macro_f1_ci95']]} recall {ev['recall']:.3f} prec {ev['precision']:.3f}")
    return jp


def run(project, seed, n, root=None, cache_root=None, order=0) -> Path:
    d = _out(root) / project
    t = tag(seed, n, order)
    jp = d / f"{t}.json"
    if jp.exists():
        return jp
    for j in range(FOLDS):
        if not (d / f"{t}_fold{j}.json").exists():
            run_fold(project, seed, n, j, root, order=order)
    return run_final(project, seed, n, root, cache_root, order=order)


def lwf_tag(seed, n, kd_weight, order=0) -> str:
    lam = f"{kd_weight:g}".replace(".", "p")
    return f"LWF_s{seed}_l{lam}_N{n}" if order == 0 else f"LWF_s{seed}_l{lam}_o{order}_N{n}"


def run_lwf_final(project, seed, n, kd_weight, root=None, cache_root=None, log=None, order=0) -> Path:
    """Learning-without-forgetting variant of the full fine-tuning final refit: same (lr, epochs) the
    plain cell already picked (its own fold search stays as is), with the stock-decoder distillation
    loss added at kd_weight. Needs that cell's json (run() or run_final() first)."""
    from ..backends.pmt import evaluate as E

    log = log or (lambda s: print(s, flush=True))
    d = _out(root) / project
    t = lwf_tag(seed, n, kd_weight, order)
    jp = d / f"{t}.json"
    if jp.exists():
        log(f"{jp} exists, kept")
        return jp
    c5f = json.loads((d / f"{tag(seed, n, order)}.json").read_text())
    lr, epochs = c5f["picked"]
    recs = data.load_split(project, "train", n=n, seed=seed)
    ev_recs = data.load_split(project, "eval")
    feats, ev_feats = _features(project, [recs, ev_recs], gt=[True, False])
    c2h = c2_head_for(project, seed, n, Images(project, recs), root)
    base = {k: v.cuda() for k, v in H.load_head_weights("cpu").items()}
    t0 = time.time()
    p = train(project, feats, range(len(recs)), base, c2h, lr, [epochs], seed=order, log=log,
              kd_weight=kd_weight)[epochs]
    tfit = time.time() - t0
    src = t.lower()
    enc = Path(cache_root) if cache_root else d / f"{t}_evalcache"
    for rec, e in zip(ev_recs, encode_all(ev_feats, range(len(ev_recs)), p)):
        C.save(enc, rec["image_key"], e)
    ev = E.evaluate(project, head_of(project, p), ref=start(project), source=src, root=enc)
    torch.save({k: v.cpu() for k, v in p.items()}, d / f"{t}.decoder.pt")
    rec = {"method": "LWF", "project": project, "seed": seed, "n": n, "order": order, "kd_weight": kd_weight,
           "picked_from_c5f": [lr, epochs], "seconds_fit": round(tfit, 1),
           "note": "lr/epochs reused from the cell's C5f pick; only kd_weight is this method's own grid",
           "eval": {k: v for k, v in ev.items() if k != "counts"}, "eval_counts": ev["counts"]}
    jp = d / f"{t}.json"
    jp.write_text(json.dumps(rec, indent=1))
    log(f"LWF {project} s{seed} o{order} N{n} lambda={kd_weight}: eval F1 {ev['macro_f1']:.4f} "
        f"{[round(x, 3) for x in ev['macro_f1_ci95']]} recall {ev['recall']:.3f} prec {ev['precision']:.3f}")
    return jp
