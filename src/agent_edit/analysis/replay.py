"""Recipe replay: run a finished run's recipe.py on another (project, seed, N) train pool, with no
agent, then score the head it gives on that project's hidden eval set.

    agent-edit replay RUN [--project P] [--seed S] [--n N] [--out F]   -> RUN/replay/<project>_s<S>_N<N>.json

With no target given it replays on the run's own pool, which must give back the saved edits.
The recipe runs in the same sandbox as at final; `train.read` runs each read on the target pool,
and a read of all the run's images reads all of the target's.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..backends.pmt.backend import FG_PARENT_COCO_ROW
from ..config import PKG, has_module
from ..harness.cli import load_backend
from ..harness.files import now
from ..harness.names import Names
from ..harness.recipe import LiveTrain, compare, run_recipe
from ..harness.server import RECIPE_TIMEOUT


def _eval_pmt(project, weights, backend):
    from ..backends.pmt.evaluate import evaluate

    return evaluate(project, weights, source=backend.source)


def _eval_clip(project, weights, backend):
    from ..backends.clip.evaluate import evaluate

    return evaluate(project, weights)


def _eval_fake(project, weights, backend):
    return backend.evaluate(weights)


# Keys are module paths below the package root: a run folder stores its backend under the root the
# package had when it ran, so only the part after the root is comparable.
SCORERS = {"backends.pmt": _eval_pmt, "backends.fake": _eval_fake}
if has_module("backends.clip.evaluate"):
    SCORERS["backends.clip"] = _eval_clip


def _tail(spec: str) -> str:
    """'root.backends.pmt.backend:make' -> 'backends.pmt.backend'."""
    return spec.partition(":")[0].partition(".")[2]


def _under(tail: str, prefix: str) -> bool:
    return tail == prefix or tail.startswith(prefix + ".")


def own_spec(spec: str) -> str:
    """A stored 'module:factory' with its root swapped for this package's name, when the module
    below the root exists here; any other spec is returned as it is."""
    mod, sep, fac = spec.partition(":")
    root, _, tail = mod.partition(".")
    if root == PKG or not tail or not has_module(tail):
        return spec
    return f"{PKG}.{tail}{sep}{fac}"


def _scorer(spec: str):
    tail = _tail(spec)
    for prefix, fn in SCORERS.items():
        if _under(tail, prefix):
            return fn
    raise SystemExit(f"replay has no eval scorer for backend {spec}")


def _n_train(backend, default) -> int:
    n = getattr(backend, "n_images", None)
    return int(n()) if callable(n) else int(default)


# The 4 fine-grained "-10" PMT projects aren't in PB.COCO_ROWS, so start="coco" PB.start_weights
# falls back to the first 10 COCO class-head rows for all of them -- same 10 rows regardless of
# project. A cars recipe that reads weights["class_rows.weight"][2] gets COCO "car" there only
# because car is COCO row 2 and 2 < 10; an aircraft recipe reading row 4 gets "airplane" the same
# way. FG_PARENT_COCO_ROW (backend.py) gives the row (continuous CLASS_MAPPING index) each
# project's own object actually is.


def _fg_parent_fix(backend_spec: str, source_project: str, target_project: str, weights: dict) -> dict:
    """Cross-set replay only: put the target project's own COCO row at whichever slot the source
    recipe's own project happens to share with it (cars: row 2, aircraft: row 4, ...), so a row a recipe
    reads as "the real object" stays that on another set instead of turning into an unrelated
    COCO class. No-op for same-project replay and for projects without a parent row in
    FG_PARENT_COCO_ROW, so it can't change what an existing recipe reproduces."""
    if source_project == target_project or not _under(_tail(backend_spec), "backends.pmt"):
        return weights
    slot, parent = FG_PARENT_COCO_ROW.get(source_project), FG_PARENT_COCO_ROW.get(target_project)
    w = weights.get("class_rows.weight")
    if slot is None or parent is None or w is None or slot >= w.shape[0] - 1:
        return weights
    from ..backends.pmt import head as H

    coco = H.load_head_weights()
    out = dict(weights)
    out["class_rows.weight"] = w.clone()
    out["class_rows.weight"][slot] = coco["class_head.weight"][parent]
    if "class_rows.bias" in weights:
        out["class_rows.bias"] = weights["class_rows.bias"].clone()
        out["class_rows.bias"][slot] = coco["class_head.bias"][parent]
    return out


def replay(run_dir, project=None, seed=None, n=None, out=None, with_eval=True) -> dict:
    run = Path(run_dir).resolve()
    h = json.loads((run / "harness.json").read_text())
    if not h["flags"].get("recipe"):
        raise SystemExit(f"{run} was not served with --recipe")
    src = run / "recipe.py" if (run / "recipe.py").exists() else run / "work" / "recipe.py"
    if not src.exists():
        raise SystemExit(f"{run} has no recipe.py")
    args = {k: v for k, v in h.get("backend_args", {}).items() if k != "overlay_dir"}
    own = {"project": args.get("project", h.get("project")), "seed": args.get("seed"), "n_shot": args.get("n_shot")}
    for k, v in (("project", project), ("seed", seed), ("n_shot", n)):
        if v is not None:
            args[k] = str(v)
    target = {"project": args.get("project", h.get("project")), "seed": args.get("seed"), "n_shot": args.get("n_shot")}
    same_pool = target == own
    out = Path(out) if out else run / "replay" / f"{target['project']}_s{target['seed']}_N{target['n_shot']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    backend = load_backend(own_spec(h["backend"]), [f"{k}={v}" for k, v in args.items()])
    names = None
    if h["flags"].get("hide_names"):
        names = Names(backend.project_info().get("classes", []), secret=backend.secret_words())
    elif h["flags"].get("permute_names"):
        # Reuse the exact mapping the run started with (harness.json "names"), not a fresh
        # derangement: the seed alone isn't enough once the run's own classes list has changed
        # (e.g. replaying on another project), and the stored mapping is already the ground truth.
        names = Names(backend.project_info().get("classes", []), secret=backend.secret_words(),
                      shown=h.get("names"))
    if names:
        backend.class_labels = dict(names.shown)
    orig = {k: v.detach().cpu().float().contiguous().clone() for k, v in backend.initial_weights().items()}
    orig = _fg_parent_fix(h["backend"], own["project"], target["project"], orig)
    source = src.read_text()
    rec = {"source_run": str(run), "recipe": str(src), "backend": h["backend"], "target": target,
           "same_pool": same_pool, "started": now()}
    N = _n_train(backend, h.get("max_images", 256))
    src_n = h.get("pool_images") or h.get("max_images")
    rec["n_images_map"] = None if same_pool else [src_n, N]
    tmp = out.parent / f".tmp_{out.stem}"
    t0 = time.time()
    try:
        train = LiveTrain(backend, orig, tmp / "reads", names, all_images=rec["n_images_map"])
        r = run_recipe(source, train, orig, tmp / "sandbox", RECIPE_TIMEOUT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["recipe_seconds"] = r["seconds"]
    rec["ok"] = r["ok"]
    if not r["ok"]:
        rec["error"] = r["error"]
        out.write_text(json.dumps(rec, indent=1))
        print(f"replay {run.name} -> {target}: recipe failed\n{r['error']}")
        return rec
    edits = r["edits"]
    weights = {**orig, **edits}
    rec["edited_tensors"] = sorted(k for k, t in edits.items() if not bool((t == orig[k]).all()))
    rec["changed_weights"] = int(sum(int((t != orig[k]).sum()) for k, t in edits.items()))
    if same_pool:
        from ..edits import load_edit

        c = compare(edits, load_edit(run), orig)
        rec["matches_saved_edit"] = {k: c[k] for k in ("reproduces", "max_abs_diff", "per_tensor")}
    rec["train"] = {"n_images": N, "original": backend.generate(None, N)["macro_f1"],
                    "edited": backend.generate(weights, N)["macro_f1"]}
    if with_eval:
        rec["eval"] = _scorer(h["backend"])(target["project"], weights, backend)
    rec["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1))
    e = rec.get("eval", {})
    print(json.dumps({"run": run.name, "target": target, "train": rec["train"],
                      "eval_f1": e.get("macro_f1"), "eval_ref_f1": e.get("ref_macro_f1"),
                      "matches_saved_edit": rec.get("matches_saved_edit", {}).get("reproduces")}))
    return rec
