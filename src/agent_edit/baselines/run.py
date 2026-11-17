"""Run one PMT baseline, score it on the hidden eval set, save weights + JSON.

    agent-edit baselines METHOD PROJECT [--seed S] [--n N] [--out-dir D]

METHOD: c2 linear probe, c5r cosine head (rows trained with Adam), c5f full decoder fine-tuning,
lwf fine-tuning with a learning-without-forgetting term.

Output: <out-dir>/<project>/<METHOD>_s<S>_N<N>.{pt,json}, default out-dir $AGENT_EDIT_RESULTS/pmt_fg/baselines.
A finished cell is not rerun.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import torch

from .. import data
from ..config import has_module
from .common import GRIDS, Images, cv_pick, fast_f1, out_dir

# (kind, module, fit fn, fold_counts fn). A kind whose module is not in the tree is left out of
# FITS, so its name is not accepted.
KINDS = (
    ("c2", "probe", "fit_c2", None),
    ("c5r", "adam", "fit_c5r", "c5r_fold_counts"),
)
# kind -> the kind whose pick its fit reads from ctx["<kind>_cfg"]; run first when not asked for
NEEDS: dict[str, str] = {}

if has_module("baselines.run_lab"):
    from . import run_lab

    KINDS, NEEDS, GRIDS = run_lab.KINDS, run_lab.NEEDS, {**GRIDS, **run_lab.GRIDS}

_MODS = {m: importlib.import_module(f"{__package__}.{m}") for m in dict.fromkeys(k[1] for k in KINDS)
         if has_module(f"baselines.{m}")}

# METHOD -> (fit(ims, cfg, ctx) -> (weights, info), fold_counts or None)
FITS = {k.upper(): (getattr(_MODS[m], f), getattr(_MODS[m], fc) if fc else None) for k, m, f, fc in KINDS if m in _MODS}
CV_METHODS = (*(k.lower() for k in FITS), *(("c5f", "lwf") if has_module("baselines.decoder_ft") else ()))


def run_cell(project, seed, n, methods, out_root=None, log=print) -> list[Path]:
    """Pick on 5 folds of the train images, refit on all of them, score once on eval."""
    from ..backends.pmt import evaluate as E

    recs = data.load_split(project, "train", n=n, seed=seed)
    ims = Images(project, recs)
    ctx = {"seed": seed}
    out = Path(out_root or out_dir()) / project
    out.mkdir(parents=True, exist_ok=True)
    for m, needed in NEEDS.items():
        if m in methods and needed not in methods:
            methods = [needed, *methods]
    done = []
    for m in methods:
        tag = f"{m}_s{seed}_N{n}"
        jp = out / f"{tag}.json"
        done.append(jp)
        if jp.exists():
            log(f"{jp} exists, kept")
            picked = json.loads(jp.read_text())["picked"]
            ctx[f"{m.lower()}_cfg"] = tuple(picked) if isinstance(picked, list) else picked
            continue
        fit, fold_counts = FITS[m]
        t0 = time.time()
        cfg, cv = cv_pick(GRIDS[m], fit, ims, ctx, fold_counts)
        ctx[f"{m.lower()}_cfg"] = cfg
        w, info = fit(ims, cfg, ctx)
        tfit = time.time() - t0
        ev = E.evaluate(project, w)
        torch.save(w, out / f"{tag}.pt")
        rec = {"method": m, "project": project, "seed": seed, "n": n, "n_train_images": len(ims),
               "picked": list(cfg) if isinstance(cfg, tuple) else cfg, "cv_f1": cv, "fit": info,
               "train_f1_fast": fast_f1(ims, w), "seconds_fit": round(tfit, 1),
               "eval": {k: v for k, v in ev.items() if k != "counts"}, "eval_counts": ev["counts"]}
        jp.write_text(json.dumps(rec, indent=1))
        log(f"{m} {project} s{seed} N{n}: picked {cfg} cv {max(cv.values()):.3f} | eval F1 {ev['macro_f1']:.4f} "
            f"{[round(x, 3) for x in ev['macro_f1_ci95']]} recall {ev['recall']:.3f} prec {ev['precision']:.3f} "
            f"({tfit:.0f}s)")
    return done


def main(method, project, seed=0, n=data.N_TRAIN, out_dir=None, fold=None, order=0, kd_weight=None):
    if method not in CV_METHODS:
        raise SystemExit(f"{method} has no cv version; cv methods: {', '.join(CV_METHODS)}")
    if order and method not in ("c5f", "lwf"):
        raise SystemExit("--order is for c5f/lwf (the batch-shuffle seed); other methods don't have one")
    if method == "c5f":
        from . import decoder_ft

        if kd_weight is not None:
            raise SystemExit("--kd-weight is for lwf")
        if fold is not None:
            return decoder_ft.run_fold(project, seed, n, fold, out_dir, order=order)
        return decoder_ft.run(project, seed, n, out_dir, order=order)
    if method == "lwf":
        from . import decoder_ft

        if fold is not None:
            raise SystemExit("--fold is for c5f; lwf reuses c5f's picked (lr, epochs) directly")
        if kd_weight is None:
            raise SystemExit("lwf needs --kd-weight (its own grid knob; try one of decoder_ft.LWF_LAMBDAS)")
        return decoder_ft.run_lwf_final(project, seed, n, kd_weight, out_dir, order=order)
    if fold is not None:
        raise SystemExit("--fold is for c5f")
    if kd_weight is not None:
        raise SystemExit("--kd-weight is for lwf")
    return run_cell(project, seed, n, [method.upper()], out_dir, log=lambda s: print(s, flush=True))
