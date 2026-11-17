"""Retention: do appended class rows break PMT on COCO?

Append the edited K rows to the stock 80 rows, keep the edited no-object row, and score COCO val2017
mask mAP from the cache at $AGENT_EDIT_PMT_ROOT/cache/coco_val2017 (annotations under $AGENT_EDIT_COCO_ROOT).
`build_cache` writes that cache (tools/build_coco_guard_cache.py); the in-run guard reads only the
first RETAIN_IMAGES ids.

    agent-edit retention RUN [RUN ...]   -> RUN/e2.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .. import config


def coco_ann() -> Path:
    return config.coco_root() / "annotations" / "instances_val2017.json"


def guard_root() -> Path:
    from ..backends.pmt import cache as C

    return C.cache_root() / "coco_val2017"


def build_cache(limit=None, root=None, device="cuda", ann=None, images=None) -> list:
    """Cache PMT's per-image output for the first `limit` COCO val2017 image ids, sorted (all when
    None), as <root>/<id>.pt. CocoVal reads the same sorted order, so the same `limit` picks the
    same images. Cached ids are skipped. Returns the ids written."""
    from pycocotools.coco import COCO

    from .. import data
    from ..backends.pmt import cache as C
    from ..backends.pmt import head as H

    coco = COCO(str(ann or coco_ann()))
    ids = sorted(coco.getImgIds())[:limit]
    root = Path(root) if root else guard_root()
    images = Path(images) if images else config.coco_root() / "val2017"
    todo = [i for i in ids if not C.path(root, i).exists()]
    print(f"[guard_cache] {len(ids)} images, {len(todo)} to cache", flush=True)
    if not todo:
        return []
    bb, dec, w = H.Backbone(device), H.build_decoder(device), H.load_head_weights(device)
    for i in todo:
        img = data.open_rgb(images / coco.imgs[i]["file_name"])
        f, rope = bb.features(img)
        C.save(root, i, C.encode(dec, w, f, rope, img.shape[:2]))
    return todo


class CocoVal:
    """COCO val2017 GT and cached entries, loaded once and reused for every head."""

    def __init__(self, limit=None, root=None):
        from pycocotools.coco import COCO

        from ..backends.pmt import cache as C

        self.coco = COCO(str(coco_ann()))
        self.ids = sorted(self.coco.getImgIds())[:limit]
        self.cats = sorted(self.coco.getCatIds())
        root = Path(root) if root else guard_root()
        missing = [i for i in self.ids if not C.path(root, i).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(self.ids)} COCO guard entries are not in {root} "
                                    f"(first: {C.path(root, missing[0]).name}); build them with tools/build_coco_guard_cache.py")
        self.entries = {i: C.load(root, i) for i in self.ids}

    def map(self, W, B, row_cats) -> float:
        """Mask mAP x100 for a class head with rows W [R, EMBED], B [R] (last row = no object).
        row_cats[r] = COCO category id of row r, or None for rows that are not COCO classes (their
        predictions still compete for the top 100 but are not scored)."""
        from pycocotools.cocoeval import COCOeval

        from ..backends.pmt.predict import predict

        dets = []
        for i in self.ids:
            p = predict(self.entries[i], W, B, threshold=0.0)
            dets += [{"image_id": i, "category_id": row_cats[lab], "segmentation": r, "score": s}
                     for r, lab, s in zip(p["rles"], p["labels"].tolist(), p["scores"].tolist()) if row_cats[lab] is not None]
        if not dets:  # a head that scores no COCO class on any image: pycocotools' loadRes([]) raises
            return 0.0
        ev = COCOeval(self.coco, self.coco.loadRes(dets), "segm")
        ev.params.imgIds = self.ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return float(ev.stats[0] * 100)


def appended_head(project, edited: dict, stock: dict, cats: list):
    """(W, B, row_cats) of the stock 80 rows + the edited K rows + the edited no-object row.
    For COCO-class projects the appended rows count as their COCO class; otherwise they are unscored."""
    from ..backends.pmt import backend as PB
    from ..backends.pmt import head as H

    Wk, Bk = edited["class_rows.weight"][:-1], edited["class_rows.bias"][:-1]
    W = torch.cat([stock["class_head.weight"][:H.NUM_CLASSES], Wk, edited["class_rows.weight"][-1:]])
    B = torch.cat([stock["class_head.bias"][:H.NUM_CLASSES], Bk, edited["class_rows.bias"][-1:]])
    extra = [cats[r] for r in PB.slot_rows(project)] if project in PB.COCO_ROWS else [None] * len(Wk)
    return W, B, cats[:H.NUM_CLASSES] + extra + [None]


def run(run_dirs, limit=None) -> list[dict]:
    """Retention for many runs: COCO entries and the stock score are computed once."""
    from ..backends.pmt import backend as PB
    from ..backends.pmt import head as H
    from ..edits import load_edit, merge

    val = CocoVal(limit)
    stock = H.load_head_weights()
    before = val.map(stock["class_head.weight"], stock["class_head.bias"], val.cats + [None])
    out = []
    for r in run_dirs:
        run_dir = Path(r).resolve()
        project = run_dir.parent.name
        ed = merge(PB.start_weights(project), load_edit(run_dir))
        after = val.map(*appended_head(project, ed, stock, val.cats))
        res = {"coco_map_before": before, "coco_map_after": after, "drop": before - after}
        (run_dir / "e2.json").write_text(json.dumps(res, indent=1))
        print("E2", project, run_dir.name, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
        out.append(res)
    return out


if config.has_module("analysis.retention_lab"):  # names of an optional sibling module, when the tree has it
    from .retention_lab import *  # noqa: F403
