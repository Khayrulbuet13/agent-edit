"""Hidden-eval scoring for PMT class-head weights (agent edits and the baselines).

    agent-edit score <project>/pmt_s<S>_N<N>_B<B>_<agent>[_ovl] [--out F]   -> eval.json, edited vs stock

The matcher, macro-F1 and paired bootstrap are those of `metrics/`. Adds recall, recall over the
mask-recall ceiling (when a ceiling file exists), and mask AP. Eval images enter only here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ... import config, data
from ...edits import load_edit, merge
from ...metrics.bootstrap import bootstrap_diff, f1_ci
from ...metrics.coco_map import mask_ap
from ...metrics.f1 import f1_from_counts, summarize
from . import backend as PB
from . import cache as C

_ENTRIES: dict = {}


def _entry(root, key):
    # every evaluate() reads the same eval images (edited and reference, ablations, random draws);
    # a tiled entry takes tens of seconds to load
    if (root, key) in _ENTRIES:
        return _ENTRIES[(root, key)]
    e = C.load(root, key)
    if "tiles" not in e:  # tiled entries are up to ~0.4 GB each in memory: stream those
        _ENTRIES[(root, key)] = e
    return e


def docs_for(project, weights, split="eval", source=None, n=None, seed=0, thr=0.0, root=None):
    source = source or PB.SOURCES.get(project, "pmt")
    root = root or C.source_root(project, source)
    recs = data.load_split(project, split, n=n, seed=seed)
    table = data.class_table(project)
    cids = [c["class_id"] for c in table]
    names = {c["class_id"]: c["class_name"] for c in table}
    box = project in PB.BOX_PROJECTS
    docs = {}
    for rec in recs:
        e = _entry(root, rec["image_key"])
        p = PB.predictions(e, weights, thr, box)
        docs[rec["image_key"]] = PB.to_doc(project, rec, e["hw"], p, names, cids, split)
    return recs, docs


def ceiling(project, source=None) -> float | None:
    """Mask-recall ceiling of the eval split from $AGENT_EDIT_RESULTS/pmt_a2, or None when there is no file."""
    source = source or PB.SOURCES.get(project, "pmt")
    if not os.environ.get("AGENT_EDIT_RESULTS"):
        return None
    p = config.results_dir() / "pmt_a2" / f"{project}__{source}__eval.json"
    return json.loads(p.read_text())["ceiling"] if p.exists() else None


def evaluate(project, weights, ref=None, source=None, start="coco", thr=PB.SCORE_THRESHOLD, with_ap=True,
            root=None) -> dict:
    """Score `weights` on the hidden eval set; `ref` (default: the stock rows for `start`) gives the
    paired diff. `start` must match what the run was served with, so the stock reference is the
    same head the agent started from (see PB.start_weights). root: a cache folder other than the
    source's own (the baselines that re-encode the eval images)."""
    source = source or PB.SOURCES.get(project, "pmt")
    recs, docs = docs_for(project, weights, source=source, root=root)
    s = summarize(project, recs, docs, thr)
    Cn = s.pop("_counts")
    tp, ng = int(Cn[..., 0].sum()), int(Cn[..., 2].sum())
    ceil = ceiling(project, source)
    out = {"project": project, "source": source, "macro_f1": s["macro_f1"], "macro_f1_ci95": f1_ci(Cn),
           "recall": tp / max(ng, 1), "precision": tp / max(int(Cn[..., :2].sum()), 1), "ceiling": ceil,
           "recall_over_ceiling": (tp / max(ng, 1)) / ceil if ceil else None,
           "per_class": s["per_class"], "n_images": s["n_images"], "n_pred": s["n_pred"],
           "counts": Cn.tolist()}
    if with_ap:
        out["mask_ap"] = mask_ap(recs, docs, [c["class_id"] for c in data.class_table(project)])
    ref = ref if ref is not None else PB.start_weights(project, source, start)
    rr, rd = docs_for(project, ref, source=source, root=root)
    Cr = summarize(project, rr, rd, thr)["_counts"]
    out["ref_macro_f1"] = f1_from_counts(Cr)[0]
    out["bootstrap_minus_ref"] = bootstrap_diff(Cr, Cn)
    return out


def parse_run_name(name: str) -> tuple[int, int]:
    m = re.match(r"pmt_s(\d+)_N(\d+)_B(\d+)", name)
    if not m:
        raise ValueError(f"not a PMT run name: {name} (want pmt_s<S>_N<N>_B<B>_...)")
    return int(m.group(1)), int(m.group(2))


def score_run(run_dir, out_path=None) -> dict:
    run = Path(run_dir).resolve()
    project = run.parent.name
    seed, n = parse_run_name(run.name)
    # source/start come from the run's own harness.json, so a parent-started run is scored
    # against the head it actually started from, not today's default.
    args = json.loads((run / "harness.json").read_text()).get("backend_args", {})
    source, start = args.get("source"), args.get("start", "coco")
    stock = PB.start_weights(project, source, start)
    edited = merge(stock, load_edit(run)) if (run / "edit" / "manifest.json").exists() else stock
    out = {"project": project, "run": str(run), "seed": seed, "n": n,
           "eval": evaluate(project, edited, source=source, start=start)}
    be = PB.make_backend(project, seed=seed, n_shot=n, source=source, start=start)
    be._ensure()
    out["train"] = {"original": be.generate(stock, len(be._recs))["macro_f1"],
                    "edited": be.generate(edited, len(be._recs))["macro_f1"]}
    dest = Path(out_path) if out_path else run / "eval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    e = out["eval"]
    print(json.dumps({"project": project, "eval_f1": e["macro_f1"], "stock_f1": e["ref_macro_f1"],
                      "diff_ci95": e["bootstrap_minus_ref"]["ci95"], "recall_over_ceiling": e["recall_over_ceiling"],
                      "train": out["train"]}))
    return out
