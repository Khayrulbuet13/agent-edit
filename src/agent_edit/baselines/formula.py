"""The Agent formula: a gradient-free edit of PMT's class rows from a few labelled images, plus the
COCO guard that picks among its settings.

Every class c present in the train images gets a row and a bias

    w_c = w_base + t_gate * v + s_cls * D_c
    b_c = b_base - t_gate * th + s_cls * bD_c

`v`/`th` are a gate direction and threshold that separate one "primary" query per object from the
other queries (duplicates and background). `D_c` is the class-mean direction: the mean embedding of
class c's primary queries, centred over classes, with `k_pcs` within-class nuisance directions
projected out; `bD_c` makes `D_c` score as a nearest-mean rule around the pooled primary mean. The
gate comes in two kinds:

  stock    the base row is a stock COCO class row (PARENT_ROW); the primary query is the one with the
           best stock score among queries that hit the object (IoU >= 0.5); the others are the
           stock-confident queries (score >= 0.5) that miss. `v` is a unit vector.
  learned  the base row is the no-object row; the primary query has the lowest no-object logit among
           the hits; the others are every other query. `v` is scaled by the primaries' own spread.

After the rows are built, one shift of every class bias (the cut) is picked by F1 on the fit images.
`(gate kind, t_gate, s_cls, k_pcs)` is chosen by 5-fold cross-validation over a fixed grid. The guard
then walks the grid best-first and takes the first setting whose COCO mask mAP on a fixed 20-image
check drops by at most one point (GUARD_BOUND); if none does, the setting with the smallest drop wins.

    python -m agent_edit.baselines.formula PROJECT --n N --seed S --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import data
from ..backends.pmt import backend as PB
from ..backends.pmt import head as H
from .common import Images, cv_pick, fast_counts, gate_sweep, start

# Stock COCO row used as the base row of the "stock" gate. Sets that are not listed get the learned
# gate only.
PARENT_ROW = {"stanford-cars-10": 2, "aircraft-10": 4, "xview-ships-10": 8,
              "stanford-cars-10-mix5": 2, "aircraft-10-mix5": 4}
CUT_GRID = np.arange(-80.0, 80.01, 5.0)
T_GATE_GRID = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0)
S_CLS_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
K_PCS_GRID = (0, 5, 10, 20, 30, 40, 50)
GUARD_BOUND = 1.0  # COCO mask mAP points


def gate_options(project: str) -> tuple[str, ...]:
    return ("stock", "learned") if project in PARENT_ROW else ("learned",)


def grid_for(project: str) -> list[tuple[str, float, float, int]]:
    return [(opt, tg, sc, kp) for opt in gate_options(project)
            for tg in T_GATE_GRID for sc in S_CLS_GRID for kp in K_PCS_GRID]


def _logits(ims: Images):
    # Rows are per true instance (inst_img), not per image: xview chips hold several instances.
    st = start(ims.project)
    Wst, bst = st["class_rows.weight"].double(), st["class_rows.bias"].double()
    return ims.q[ims.inst_img].double() @ Wst.T + bst, Wst, bst


def _stats(ims: Images):
    """sc: best class probability x mask score; noobj: the no-object logit; matched: IoU >= 0.5 to
    the row's own instance."""
    logits, Wst, bst = _logits(ims)
    K = ims.K
    probs = logits.softmax(-1)[..., :K]
    sc = probs.max(-1).values * ims.mscore[ims.inst_img].double()
    noobj = logits[..., K]
    matched = ims.inst_iou.double() >= 0.5
    return sc, noobj, matched, Wst, bst


def _primary(sc, noobj, matched, mode):
    prim_img, prim_q = [], []
    for i in range(matched.shape[0]):
        idx = torch.where(matched[i])[0]
        if len(idx) == 0:
            continue
        key = sc[i, idx] if mode == "stock" else -noobj[i, idx]
        prim_img.append(i)
        prim_q.append(int(idx[torch.argmax(key)]))
    return torch.tensor(prim_img, dtype=torch.long), torch.tensor(prim_q, dtype=torch.long)


def _other_mask(matched, sc, prim_img, prim_q, mode):
    if mode == "stock":
        return (~matched) & (sc >= 0.5)
    m = torch.ones_like(matched)
    m[prim_img, prim_q] = False
    return m


def _shared(ims: Images, mode: str, max_k: int):
    """What a (t_gate, s_cls, k_pcs) triple does not change: gate direction and threshold, centred
    class means, and the SVD of the within-class residual, computed once up to `max_k` directions.
    None when there are too few primary or other queries to define a gate."""
    K = ims.K
    sc, noobj, matched, Wst, bst = _stats(ims)
    prim_img, prim_q = _primary(sc, noobj, matched, mode)
    other = _other_mask(matched, sc, prim_img, prim_q, mode)
    if len(prim_img) < 2 or int(other.sum()) < 2:
        return None

    q = ims.q[ims.inst_img].double()
    Xp = q[prim_img, prim_q]
    yp = ims.inst_gt[prim_img]
    Eo = q[other]

    d = Xp.mean(0) - Eo.mean(0)
    v = d / d.norm() if mode == "stock" else d / float((Xp @ d).std())
    th = 0.5 * float((Xp @ v).mean() + (Eo @ v).mean())

    present = sorted(set(int(x) for x in yp.tolist()))
    mu = torch.zeros(K, Xp.shape[1], dtype=torch.float64)
    for c in present:
        mu[c] = Xp[yp == c].mean(0)
    mu_present = torch.stack([mu[c] for c in present])
    Dc_base = mu_present - mu_present.mean(0)
    m0 = Xp.mean(0)
    Z = Xp - mu[yp]
    # Keep the removed directions below the residual rank, or they eat the class signal.
    max_k_cap = min(max_k, max(0, (Xp.shape[0] - len(present)) // 3))
    Vh_full = torch.linalg.svd(Z, full_matrices=False)[2][:max_k_cap] if max_k_cap > 0 else None

    if mode == "stock":
        row = PARENT_ROW[ims.project]
        Wb, bb = Wst[row], float(bst[row])
    else:
        Wb, bb = Wst[K], float(bst[K])

    return dict(present=present, Dc_base=Dc_base, m0=m0, v=v, th=th, Vh_full=Vh_full,
                max_k_cap=max_k_cap, Wb=Wb, bb=bb)


def _combine(ims: Images, shared, t_gate, s_cls, k_pcs):
    """Class rows for one grid point: remove k_pcs nuisance directions from the class means, then
    add the gate and the scaled means to the base row."""
    present, Dc = shared["present"], shared["Dc_base"]
    k = min(k_pcs, shared["max_k_cap"])
    if k > 0:
        Vh = shared["Vh_full"][:k]
        Dc = Dc - (Dc @ Vh.T) @ Vh
    bD = -(Dc @ shared["m0"]) - 0.5 * (Dc ** 2).sum(1)
    w = start(ims.project)
    for j, c in enumerate(present):
        w["class_rows.weight"][c] = (shared["Wb"] + t_gate * shared["v"] + s_cls * Dc[j]).float()
        w["class_rows.bias"][c] = shared["bb"] - t_gate * shared["th"] + s_cls * float(bD[j])
    return w


def fit_formula(ims: Images, cfg):
    """cfg = (gate kind, t_gate, s_cls, k_pcs) -> (weights, info). No cut yet; see fit_formula_final."""
    gate_opt, t_gate, s_cls, k_pcs = cfg
    shared = _shared(ims, gate_opt, k_pcs)
    if shared is None:
        return start(ims.project), {"skipped": "too few primary/other queries"}
    return _combine(ims, shared, t_gate, s_cls, k_pcs), {"n_prim": len(shared["present"])}


def fit_formula_final(ims: Images, cfg):
    """fit_formula, then the class-bias cut that maximises F1 on the fit images."""
    w, info = fit_formula(ims, cfg)
    if "skipped" in info:
        return w, info
    return gate_sweep(ims, w, unit=1.0, grid=CUT_GRID, rows="classes")


def formula_fold_counts(fi: Images, hi: Images, grid, ctx):
    """One held-out count table per grid point; the per-kind `_shared` work is done once per fold."""
    by_mode: dict[str, dict | None] = {}
    out = []
    for gate_opt, t_gate, s_cls, k_pcs in grid:
        if gate_opt not in by_mode:
            max_k = max(kp for (go, _, _, kp) in grid if go == gate_opt)
            by_mode[gate_opt] = _shared(fi, gate_opt, max_k)
        shared = by_mode[gate_opt]
        if shared is None:
            out.append(fast_counts(hi, start(fi.project)))
            continue
        w = _combine(fi, shared, t_gate, s_cls, k_pcs)
        w2, _ = gate_sweep(fi, w, unit=1.0, grid=CUT_GRID, rows="classes")
        out.append(fast_counts(hi, w2))
    return out


def cv_pick_formula(ims: Images):
    """(best grid point, {str(grid point): held-out macro-F1}) by 5-fold CV on the train images."""
    return cv_pick(grid_for(ims.project), None, ims, {}, fold_counts=formula_fold_counts)


class RetentionGuard:
    """COCO mask mAP on the first `limit` val2017 images for the stock head, and for the stock head
    with a candidate's rows appended. Loaded on first use and kept for later cells."""

    def __init__(self, limit: int = PB.RETAIN_IMAGES):
        self.limit = limit
        self._ctx = None

    def _load(self):
        if self._ctx is None:
            from ..analysis.retention import CocoVal

            coco = CocoVal(limit=self.limit)
            stock = H.load_head_weights("cpu")
            with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints ~20 lines per run
                before = coco.map(stock["class_head.weight"], stock["class_head.bias"], coco.cats + [None])
            self._ctx = (coco, stock, before)
        return self._ctx

    def drop(self, project, w) -> float:
        from ..analysis.retention import appended_head

        coco, stock, before = self._load()
        W, B, row_cats = appended_head(project, w, stock, coco.cats)
        with contextlib.redirect_stdout(io.StringIO()):
            after = coco.map(W, B, row_cats)
        return round(before - after, 3)


def guard_walk(grid, cv, fit, drop, bound=GUARD_BOUND):
    """Walk `grid` best CV score first (ties keep grid order) and return the first candidate whose
    drop is within `bound`; if none is, the one with the smallest drop. fit(cfg) -> (weights, info),
    drop(weights) -> float. Returns (chosen, guard_passed, n_checked); chosen has picked, cv_f1,
    retain20_drop, weights, info."""
    order = sorted(range(len(grid)), key=lambda i: (-cv[str(grid[i])], i))
    fallback, checked = None, 0
    for gi in order:
        cfg = grid[gi]
        w, info = fit(cfg)
        d = drop(w)
        checked += 1
        row = {"picked": list(cfg), "cv_f1": cv[str(cfg)], "retain20_drop": d, "weights": w, "info": info}
        if fallback is None or d < fallback["retain20_drop"]:
            fallback = row
        if d <= bound:
            return row, True, checked
    return fallback, False, checked


def run_cell(project, n, seed, out_dir, guard: RetentionGuard | None = None, bound=GUARD_BOUND) -> dict:
    """One cell: CV-pick on the n-per-class train images, guard walk, score once on the eval set.
    Writes <out_dir>/<project>_s<seed>_N<n>.{json,pt}.

    n=1 has no usable CV fold, so it ranks the grid with the same set and seed's n=2 CV scores."""
    from ..backends.pmt import evaluate as E

    t0 = time.time()
    out_dir = Path(out_dir)
    guard = guard or RetentionGuard()
    ims = Images(project, data.load_split(project, "train", n=n, seed=seed))
    carried = None
    if n == 1:
        ims2 = Images(project, data.load_split(project, "train", n=2, seed=seed))
        _, cv = cv_pick_formula(ims2)
        carried = f"{project}_s{seed}_N2"
    else:
        _, cv = cv_pick_formula(ims)

    chosen, passed, checked = guard_walk(grid_for(project), cv, lambda cfg: fit_formula_final(ims, cfg),
                                         lambda w: guard.drop(project, w), bound)
    ev = E.evaluate(project, chosen["weights"], with_ap=False)
    rec = {"project": project, "seed": seed, "n": n, "picked": chosen["picked"], "cv_f1": chosen["cv_f1"],
           "retain20_drop": chosen["retain20_drop"], "guard_passed": passed, "n_cv_points_checked": checked,
           "eval_f1": ev["macro_f1"], "eval_ci95": ev["macro_f1_ci95"], "fit_info": chosen["info"],
           "seconds": round(time.time() - t0, 1), "cv_all": cv, "carried_from_n2": carried}
    tag = f"{project}_s{seed}_N{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    torch.save(chosen["weights"], out_dir / f"{tag}.pt")
    print(f"{tag}: picked {chosen['picked']} cv {chosen['cv_f1']:.4f} retain20_drop "
          f"{chosen['retain20_drop']:.3f} guard_passed {passed} checked {checked} "
          f"eval {rec['eval_f1']:.4f} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agent_edit formula", description="One CV-tuned, COCO-guarded formula cell.")
    ap.add_argument("project")
    ap.add_argument("--n", type=int, default=data.N_TRAIN, help="train images per class")
    ap.add_argument("--seed", type=int, default=0, help="train-pool draw")
    ap.add_argument("--out", required=True, type=Path, help="folder for <project>_s<seed>_N<n>.{json,pt}")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run_cell(a.project, a.n, a.seed, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
