"""Harness backend for PMT's class head (class rows + no-object row editable, mask path frozen).

    agent-edit serve --backend agent_edit.backends.pmt.backend:make_backend --backend-arg project=cub-10 ...

Each image is cached once (cache.py): query embeddings, per-query masks, mask scores. generate/read
then only need the class rows, so they run on CPU. Train images only; eval images and GT never
leave this module.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

from ... import config, data
from ...harness.overlays import LEGEND, draw_errors, draw_sheet, render_one, worst_k
from ...metrics import docs as md
from ...metrics.f1 import train_report
from ...metrics.match import rle_of
from ..base import Backend
from . import cache as C
from . import head as H
from .predict import class_logits, predict, predict_tiled
from .tiling import box_xywh, flat, iou_with_gt

SCORE_THRESHOLD = 0.5
IOU_MATCH = 0.5
READ_TOP_DEFAULT = 30
READ_IMAGES_DEFAULT = 3

# PMT's contiguous COCO rows (CLASS_MAPPING) for projects whose classes are COCO classes.
COCO_ROWS: dict[str, dict[str, int]] = {}

# start="parent": every slot of a fine-grained "-10" project starts as a copy of this one COCO
# row (its real parent class), the TFA-style init, instead of start="coco"'s first 10 COCO rows
# (whose meaning is unrelated to the project). Reused by analysis/replay.py for cross-set replay.
# The 4 "-mix5" sets (invented class groups) share an image set and a real-world object with their
# base "-10" set, so they share its parent row too.
FG_PARENT_COCO_ROW = {"stanford-cars-10": 2, "cub-10": 14, "pets-10": 15, "aircraft-10": 4,
                      "stanford-cars-10-mix5": 2, "cub-10-mix5": 14, "pets-10-mix5": 15, "aircraft-10-mix5": 4,
                      "xview-ships-10": 8}
FG_PARENT_NAME = {"stanford-cars-10": "car", "cub-10": "bird", "pets-10": "cat", "aircraft-10": "airplane",
                  "stanford-cars-10-mix5": "car", "cub-10-mix5": "bird", "pets-10-mix5": "cat",
                  "aircraft-10-mix5": "airplane", "xview-ships-10": "boat"}

TASKS = {
    "stanford-cars-10": "Photos of cars. Each image shows one labelled car; the 10 classes are look-alike sedan makes and models. Find the car and name its make and model. Scored with box IoU (the box around a predicted mask).",
    "cub-10": "Photos of birds. Each image shows one labelled bird; the 10 classes are look-alike gull and tern species. Find the bird and name its species. Scored with box IoU (the box around a predicted mask).",
    "pets-10": "Photos of cats. Each image shows one labelled cat; the 10 classes are look-alike cat breeds. Find the cat and name its breed. Masks are matched to true masks.",
    "aircraft-10": "Photos of airliners. Each image shows one labelled aircraft; the 10 classes are look-alike Boeing 737 and Airbus A319/A320 variants. Find the aircraft and name its variant. Scored with box IoU (the box around a predicted mask).",
    # -mix5: each set's 10 look-alike classes above, randomly paired into 5 groups that don't
    # exist in the world (invented classes). No class or species word here: the point is a
    # group the agent can only tell apart from the images, not from a name it already knows.
    "stanford-cars-10-mix5": "Photos of cars. Each image shows one labelled car; the 5 classes are groups of look-alike car models. Find the car and name its group. Scored with box IoU (the box around a predicted mask).",
    "cub-10-mix5": "Photos of birds. Each image shows one labelled bird; the 5 classes are groups of look-alike birds. Find the bird and name its group. Scored with box IoU (the box around a predicted mask).",
    "pets-10-mix5": "Photos of cats. Each image shows one labelled cat; the 5 classes are groups of look-alike cat breeds. Find the cat and name its group. Masks are matched to true masks.",
    "aircraft-10-mix5": "Photos of airliners. Each image shows one labelled aircraft; the 5 classes are groups of look-alike airliners. Find the aircraft and name its group. Scored with box IoU (the box around a predicted mask).",
    "xview-ships-10": "Overhead satellite images. A chip can show more than one vessel; the 10 classes are "
                      "look-alike vessel types, from small boats to large ships. Find every vessel and name its "
                      "type. Scored with box IoU (the box around a predicted mask).",
}
BOX_PROJECTS = {"stanford-cars-10", "cub-10", "aircraft-10", "stanford-cars-10-mix5", "cub-10-mix5",
                "aircraft-10-mix5", "xview-ships-10"}

MODEL_INTRO = ("You are given the class head of PMT, an instance segmentation model (a frozen DINOv3 image encoder "
               "and a mask decoder) that returns a mask, a class and a score for each object it finds.")

# Optional module with more datasets (tiled or SAM-mask sources, extra task texts); used when present.
if importlib.util.find_spec(".lab_sets", __package__):
    from . import lab_sets as LAB

    COCO_ROWS.update(LAB.COCO_ROWS)
    TASKS.update(LAB.TASKS)
    BOX_PROJECTS |= LAB.BOX_PROJECTS
else:
    LAB = None

# Mask source per project; a project not listed gets one PMT pass per image.
SOURCES: dict[str, str] = {} if LAB is None else dict(LAB.SOURCES)

RETAIN_IMAGES = 20
RETAIN_ABOUT = ("COCO mask mAP (x100) on {n} fixed COCO val2017 images, for the model's original 80-class COCO head with "
                "your class rows added to it and its no-object row replaced by yours. It shows how much your rows take "
                "over objects the original model already finds.")


def _lab(source: str):
    if LAB is None:
        raise RuntimeError(f"mask source {source!r} needs lab_sets.py, which this tree does not have")
    return LAB


def make_backend(project: str, threshold: float = SCORE_THRESHOLD, seed=None, n_shot=None, source=None,
                 overlay_dir=None, start: str = "coco", **_):
    return PmtHeadBackend(project, threshold=float(threshold), seed=seed, n_shot=n_shot,
                          source=source or SOURCES.get(project, "pmt"), overlay_dir=overlay_dir, start=start)


def slot_rows(project: str) -> list[int]:
    """Class-head row for each project class, in data.class_table order."""
    names = [c["class_name"] for c in data.class_table(project)]
    if project in COCO_ROWS:
        return [COCO_ROWS[project][n] for n in names]
    return list(range(len(names)))


def parent_slot0(project: str, coco_rows: list[int]) -> int | None:
    """start='parent': the COCO row slot 0 should copy, or None if start='parent' is a no-op for
    this project. `coco_rows` is slot_rows(project) (start='coco''s rows, no no-object). If the
    project's real COCO parent class (FG_PARENT_COCO_ROW) already sits in that slice -- cars and
    aircraft, where it only landed there by the first-10-COCO-rows coincidence -- there is nothing
    to change: start='parent' gives the same head as start='coco'. Otherwise (cub, pets) the parent
    row doesn't appear anywhere in the slice, so it replaces slot 0."""
    parent = FG_PARENT_COCO_ROW.get(project)
    if parent is None:
        raise ValueError(f"start='parent' has no COCO parent row for project {project!r}")
    return None if parent in coco_rows else parent


def start_weights(project: str, source: str | None = None, start: str = "coco") -> dict:
    """The stock rows for this project's slots + the no-object row (last).

    start="coco" (default): the project's own COCO rows
    (COCO_ROWS) if it has any, else the first K COCO rows. start="parent": slot 0 copies the one
    COCO row for the project's real parent class (FG_PARENT_COCO_ROW) instead of whatever
    start='coco' put there -- the same condition the cars project already gets by coincidence
    (COCO "car" sits inside its first-10-rows slice) -- and is a no-op where that coincidence
    already holds (cars, aircraft). Slots 1..K-1 and the no-object row are start='coco''s, so the
    agent still starts from a head that can't yet tell the classes apart, not a blank one.

    An AMG source has no stock head (its features are pooled DINOv3 tokens), so it starts blank
    regardless of `start`: zero rows and a no-object bias that keeps every mask out."""
    source = source or SOURCES.get(project, "pmt")
    coco = slot_rows(project)
    K = len(coco)
    if source.startswith("amg"):
        return _lab(source).blank_head(K)
    w = H.load_head_weights()
    if start == "coco":
        rows = coco + [H.NO_OBJECT]
    elif start == "parent":
        slot0 = parent_slot0(project, coco)
        rows = coco + [H.NO_OBJECT]
        if slot0 is not None:
            rows = [slot0] + rows[1:]
    else:
        raise ValueError(f"unknown start={start!r}; use 'coco' or 'parent'")
    return {"class_rows.weight": w["class_head.weight"][rows].clone(),
            "class_rows.bias": w["class_head.bias"][rows].clone()}


def box_rle(rle: dict) -> dict:
    """Filled box of a mask, so mask IoU between two box RLEs equals box IoU."""
    h, w = rle["size"]
    x, y, bw, bh = mask_utils.toBbox(rle).tolist()
    r = mask_utils.frPyObjects(np.array([[x, y, bw, bh]], dtype=np.float64), h, w)[0]
    return {"size": [h, w], "counts": r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]}


def extra_logit_fn(weights: dict):
    """Logit terms a linear head cannot express, or None: the Tip-Adapter cache baseline
    (alpha * exp(-beta (1 - cos(q, key))) @ one-hot values) and the cosine-head baseline
    (cos_scale * cos(q, cos_rows))."""
    fns = []
    if "tip_keys" in weights:
        K, V = weights["tip_keys"], weights["tip_vals"]
        a, bt = float(weights["tip_alpha"]), float(weights["tip_beta"])

        def tip(q):
            qn = q / q.norm(dim=1, keepdim=True).clamp(min=1e-6)
            return a * torch.exp(-bt * (1 - qn @ K.T)) @ V
        fns.append(tip)
    if "cos_rows" in weights:
        Wn = weights["cos_rows"] / weights["cos_rows"].norm(dim=1, keepdim=True).clamp(min=1e-6)
        s = float(weights["cos_scale"])
        fns.append(lambda q: s * (q / q.norm(dim=1, keepdim=True).clamp(min=1e-6)) @ Wn.T)
    if not fns:
        return None
    return lambda q: sum(f(q) for f in fns)


def predictions(entry: dict, weights: dict, threshold: float, box_mode: bool = False) -> dict:
    w, b = weights["class_rows.weight"], weights["class_rows.bias"]
    fn = extra_logit_fn(weights)
    p = (predict_tiled(entry, w, b, threshold=threshold, logit_fn=fn) if "tiles" in entry
         else predict(entry, w, b, threshold=threshold, logit_fn=fn))
    if box_mode:
        p["rles"] = [box_rle(r) for r in p["rles"]]
    return p


def to_doc(project, rec, hw, p, names, class_ids, split) -> dict:
    inst = []
    for r, lab, s in zip(p["rles"], p["labels"].tolist(), p["scores"].tolist()):
        x, y, bw, bh = mask_utils.toBbox(r).tolist()
        cid = class_ids[lab]
        inst.append({"class_id": cid, "class_name": names[cid], "score": float(s),
                     "box": [x, y, x + bw, y + bh], "rle": r})
    return md.make_doc(project, rec["image_key"], hw, inst, split=split, model="pmt")


class PmtHeadBackend(Backend):
    overlay_legend = LEGEND
    supports_sheet = True

    def __init__(self, project: str, threshold: float = SCORE_THRESHOLD, seed=None, n_shot=None, source="pmt",
                 overlay_dir=None, start: str = "coco"):
        self.project, self.thr = project, threshold
        self.flags_given = seed is not None or n_shot is not None
        self.seed = 0 if seed is None else int(seed)
        self.n_shot = data.N_TRAIN if n_shot is None else int(n_shot)
        self.box_mode = project in BOX_PROJECTS
        table = data.class_table(project)
        self.class_ids = [c["class_id"] for c in table]
        self.names = {c["class_id"]: c["class_name"] for c in table}
        self.source = source
        self.start = start
        self.root = C.source_root(project, source)
        self._recs = self._entries = None
        self.overlay_dir = overlay_dir
        self._n_overlays = 0
        self._coco = self._stock = None

    def _ensure(self):
        if self._recs is not None:
            return
        t0 = time.time()
        recs = data.load_split(self.project, "train", n=self.n_shot, seed=self.seed)
        missing = [r["image_key"] for r in recs if not C.path(self.root, r["image_key"]).exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} train images not cached under {self.root}; "
                               f"run `agent-edit cache pmt {self.project}` on a GPU first")
        self._entries = [C.load(self.root, r["image_key"]) for r in recs]
        self._recs = recs
        print(f"[pmt_backend] {self.project} seed {self.seed} n_shot {len(recs)} ready in {time.time() - t0:.1f}s",
              flush=True)

    def n_images(self) -> int:
        """Train images available; for per-class projects n_shot counts images per class."""
        return len(data.train_keys(self.project, self.n_shot, self.seed))

    def initial_weights(self) -> dict:
        return start_weights(self.project, self.source, self.start)

    def _w(self, weights):
        return weights if weights is not None else start_weights(self.project, self.source, self.start)

    def _subset(self, n: int) -> list:
        return list(range(max(1, min(int(n), len(self._recs)))))

    def generate(self, weights, n_images: int, which: str = "train") -> dict:
        self._ensure()
        if which != "train":
            raise ValueError("only which='train' is available")
        t0 = time.time()
        idx = self._subset(n_images)
        w = self._w(weights)
        docs = {}
        for i in idx:
            rec, e = self._recs[i], self._entries[i]
            p = predictions(e, w, self.thr, self.box_mode)
            docs[rec["image_key"]] = to_doc(self.project, rec, e["hw"], p, self.names, self.class_ids, "train")
        t_pred = time.time() - t0
        overlays = self._overlays(idx, docs) if self.overlay_dir else None
        out = train_report(self.project, [self._recs[i] for i in idx], docs, n_images, self.seed, self.thr)
        out["seconds"] = round(time.time() - t0, 2)
        out["seconds_predict"] = round(t_pred, 2)
        if overlays:
            out["overlays"] = overlays
        out["_idx"], out["_docs"] = idx, docs
        return out

    def render_errors(self, result: dict, out_dir, sheet_k: int | None = None) -> list[str]:
        idx, docs = result["_idx"], result["_docs"]
        recs = [self._recs[i] for i in idx]
        files = draw_errors(recs, idx, docs, out_dir, self.thr)
        if sheet_k:
            self._draw_sheet(result, idx, recs, docs, out_dir, sheet_k)
        return files

    def _sheet_label(self, i, rec, doc, f1) -> str:
        # gt_by_cid has every project class as a key; most lists are empty.
        gt = ", ".join(self.shown_name(self.names[c]) for c in sorted(rec["gt_by_cid"]) if rec["gt_by_cid"][c])
        top = max((d for d in doc["instances"] if d["score"] >= self.thr), key=lambda d: d["score"], default=None)
        pred = self.shown_name(top["class_name"]) if top else "none"
        return f"#{i} f1={f1:.2f} GT:{gt} pred:{pred}"

    def _draw_sheet(self, result, idx, recs, docs, out_dir, sheet_k) -> None:
        """--overlay-sheet: the sheet_k worst-scored images of this result, same drawing as the
        per-image overlays, each tile labelled with its index, GT class and top prediction."""
        f1s = result.get("per_image_f1") or [0.0] * len(idx)
        order = worst_k(f1s, sheet_k)
        tiles = []
        for j in order:
            i, rec = idx[j], recs[j]
            doc = docs[rec["image_key"]]
            img = render_one(rec, doc, self.thr)
            tiles.append((img, self._sheet_label(i, rec, doc, f1s[j])))
        draw_sheet(tiles, out_dir)

    def _overlays(self, idx, docs) -> dict:
        """The overlay_dir backend arg: pictures with every generate, also for 'original'."""
        self._n_overlays += 1
        d = Path(self.overlay_dir) / f"overlays_{self._n_overlays}"
        files = draw_errors([self._recs[i] for i in idx], idx, docs, d, self.thr)
        return {"dir": str(d), "files": files, "legend": self.overlay_legend}

    def retention(self, weights) -> dict | None:
        if self.source.startswith("amg"):
            return None  # AMG rows read pooled DINOv3 features, not PMT queries: appending them to COCO means nothing
        from ...analysis.retention import CocoVal, appended_head

        with contextlib.redirect_stdout(io.StringIO()):
            if self._coco is None:
                self._coco = CocoVal(limit=RETAIN_IMAGES)
                self._stock = H.load_head_weights()
            score = self._coco.map(*appended_head(self.project, self._w(weights), self._stock, self._coco.cats))
        return {"score": score, "metric": "coco_mask_map", "n_images": len(self._coco.ids),
                "about": RETAIN_ABOUT.format(n=len(self._coco.ids))}

    def _gt(self, rec):
        rles, cids = [], []
        for cid in sorted(rec["gt_by_cid"]):
            for r in rec["gt_by_cid"][cid]:
                rles.append(rle_of(r))
                cids.append(int(cid))
        return rles, cids

    def read(self, weights, spec: dict) -> dict:
        self._ensure()
        what = spec.get("what", "queries")
        if what != "queries":
            raise ValueError(f"unknown what={what!r}; available: 'queries'")
        t0 = time.time()
        if spec.get("images") is not None:
            idx = [int(i) for i in spec["images"]]
            bad = [i for i in idx if not 0 <= i < len(self._recs)]
            if bad:
                raise ValueError(f"image indices out of range 0..{len(self._recs) - 1}: {bad}")
        else:
            idx = self._subset(spec.get("n_images") or READ_IMAGES_DEFAULT)
        top = int(spec.get("top", READ_TOP_DEFAULT))
        if not 1 <= top <= H.NUM_Q:
            raise ValueError(f"top must be in 1..{H.NUM_Q}")
        min_score = float(spec.get("min_score", 0.0))
        w = self._w(weights)
        W, B = w["class_rows.weight"], w["class_rows.bias"]
        K = W.shape[0] - 1

        cols = {k: [] for k in ("embedding", "image_index", "query_index", "logits", "score", "pred_class",
                                "mask_score", "box", "matched_class", "matched_iou")}
        gt_counts, img_info = {}, []
        for i in idx:
            rec, e = self._recs[i], flat(self._entries[i])
            h, wd = e["hw"]
            logits = class_logits(e, W, B)
            probs = logits.softmax(-1)[:, :K]
            best, cls = probs.max(1)
            score = best * e["mscore"]
            order = score.argsort(descending=True)[:top]
            order = order[score[order] >= min_score]
            k = int(order.numel())
            gt_counts[str(i)] = {self.names[c]: len(v) for c, v in rec["gt_by_cid"].items() if len(v)}
            img_info.append({"index": i, "image_key": rec["image_key"], "hw": [h, wd]})
            if k == 0:
                continue
            q = order.tolist()
            grles, gcids = self._gt(rec)
            if grles:
                if self.box_mode:
                    iou = mask_utils.iou([box_rle(e["rles"][j]) for j in q], grles, [0] * len(grles))
                else:
                    iou = iou_with_gt(e, grles, q)[0]
                bi, bj = iou.max(1), iou.argmax(1)
                matched = np.where(bi >= IOU_MATCH, np.asarray(gcids)[bj], -1)
            else:
                bi, matched = np.zeros(k, np.float32), np.full(k, -1)
            boxes = np.array([box_xywh(e, j) for j in q], dtype=np.float64).reshape(-1, 4)
            boxes = np.stack([boxes[:, 0] / wd, boxes[:, 1] / h, (boxes[:, 0] + boxes[:, 2]) / wd,
                              (boxes[:, 1] + boxes[:, 3]) / h], 1)
            cols["embedding"].append(e["q"][order].numpy())
            cols["image_index"].append(np.full(k, i))
            cols["query_index"].append(order.numpy())
            cols["logits"].append(logits[order].numpy())
            cols["score"].append(score[order].numpy())
            cols["pred_class"].append(np.asarray(self.class_ids)[cls[order].numpy()])
            cols["mask_score"].append(e["mscore"][order].numpy())
            cols["box"].append(boxes.astype(np.float32))
            cols["matched_class"].append(matched.astype(np.int64))
            cols["matched_iou"].append(np.asarray(bi, np.float32))
        empty = {"embedding": (0, H.EMBED), "logits": (0, K + 1), "box": (0, 4)}
        arrays = {k: (np.concatenate(v) if v else np.zeros(empty.get(k, (0,)))) for k, v in cols.items()}
        return {
            **arrays,
            "n_rows": int(len(arrays["score"])),
            "columns": "all arrays have one row per (image, query), best score first within an image; logits are the "
                       f"{K + 1} class-head outputs (the {K} class rows in class order, then no-object); pred_class is the "
                       "class id with the highest probability; matched_class is the class id of the best-overlapping "
                       "true instance of any class (IoU >= 0.5), else -1; matched_iou is that best IoU",
            "classes": [{"class_id": c, "class_name": self.names[c], "row": j} for j, c in enumerate(self.class_ids)],
            "images": img_info,
            "gt_counts": gt_counts,
            "seconds": round(time.time() - t0, 2),
        }

    def frozen_patterns(self) -> list:
        return []

    def secret_words(self) -> list[str]:
        """A 'cache_from' project (the -mix5 sets) keeps its parent's image keys, which spell the
        parent's own classes (e.g. cub-10-mix5 images are still named "California_Gull_...").
        Those words aren't this project's classes, so Names() must be told about them separately."""
        parent = data.manifest(self.project).get("cache_from")
        return [c["class_name"] for c in data.class_table(parent)] if parent else []

    def project_info(self) -> dict:
        info = {"classes": [self.names[c] for c in self.class_ids], "task": TASKS.get(self.project, ""),
                "model_intro": _lab(self.source).MODEL_INTRO_AMG if self.source.startswith("amg") else MODEL_INTRO}
        if self.flags_given:
            info["n_shot"] = self.n_images()
        return info

    def describe(self) -> str:
        K = len(self.class_ids)
        cls = "; ".join(f"row {j} = class id {c} = {self.names[c]}" for j, c in enumerate(self.class_ids))
        match = ("Boxes, not masks, are matched for this project: the box around each predicted mask against the true box."
                 if self.box_mode else "Masks are matched to true masks.")
        text = DESCRIBE
        slot0 = parent_slot0(self.project, slot_rows(self.project)) if self.start == "parent" else None
        if slot0 is not None:
            start_note = START_NOTE_PARENT.format(name=FG_PARENT_NAME.get(self.project, "?"))
        else:
            start_note = START_NOTE_COCO.format(K=K)  # start="coco", or start="parent" where it's a no-op
        if self.source != "pmt":
            text = _lab(self.source).describe_text(self.source, text)
        n = self.n_images()
        return text.format(K=K, K1=K + 1, classes=cls, thr=self.thr, max_images=n,
                           top=READ_TOP_DEFAULT, n_read=READ_IMAGES_DEFAULT, match=match, max_images_m1=n - 1,
                           start_note=start_note)


START_NOTE_COCO = ("The class rows you start from are rows of its COCO class head (for COCO-like projects, the "
                   "matching COCO classes; otherwise the first {K} COCO classes, whose meaning is unrelated to "
                   "this project) and the no-object row of that head.")
START_NOTE_PARENT = ("Row 0 starts as a copy of COCO's \"{name}\" row, so the original head finds {name}s but "
                     "calls every one of them row 0's class; the other rows are rows of COCO's class head, "
                     "whose meaning is unrelated to this project.")

DESCRIBE = """\
Frozen and not editable: the DINOv3 image encoder and the whole mask decoder. For each image the decoder gives 200 queries; each query has a fixed mask, a fixed mask score (how confident the mask is, 0..1) and a fixed 1024-number embedding. You edit only the class head that turns a query embedding into class logits: 2 tensors, {K1} rows. `NAMES` lists them.
- `class_rows.weight` ({K1}, 1024) and `class_rows.bias` ({K1},). Rows 0..{K} - 1 are the project classes: {classes}. Row {K} is the "no object" class.

The model was trained on COCO (80 everyday classes). {start_note}

For one image the head does:
1. logits = embedding @ class_rows.weight.T + class_rows.bias, for each of the 200 queries: ({K1}) numbers per query.
2. probabilities = softmax over those {K1} logits (the no-object class takes part in the softmax).
3. For every (query, class) pair with class < {K}: score = probability of that class x the query's mask score. The 100 best pairs are kept (one query can appear with several classes).
4. Pairs with score >= {thr} become predictions: that query's mask, labelled with that class.
Everything runs in float32, so small edits take effect.

SCORING. `generate` runs the head on each image, and per class matches predictions to true instances at IoU 0.5 (greedy, best score first). {match} macro_f1 = per-class F1 (counts pooled over images), averaged over classes with a true instance or a prediction. per_image_f1 = F1 averaged over the classes that have a true instance or a prediction in that image (1.0 if none). There are only {max_images} training images (n_images is capped at it); n_images=k gives the first k of a fixed random order, so the same k gives the same images. A generate takes seconds.

READ. {{"type": "read", "model": "edited", "what": "queries", ...}} returns one row per (image, query) for the `top` best-scoring queries of each image. Options, all optional: "images": [0, 3] (indices 0..{max_images_m1}; default the first n_images, n_images default {n_read}), "top": 1..200 (default {top}; 200 = every query, including ones that match nothing), "min_score". Arrays (saved as .npy, one row each): embedding (R,1024) = the query vectors the class head reads; image_index; query_index (0..199); logits (R,{K1}); score (best class probability x mask score); pred_class; mask_score; box (xyxy as fractions of the image); matched_class (class id of the true instance whose mask overlaps this query's mask with IoU >= 0.5, else -1); matched_iou. Also gt_counts (true instances per class per image). matched_class not in (-1, pred_class): right object, wrong class. matched_class == -1: no true object there.
"""


if config.has_module("backends.pmt.backend_lab"):  # names of an optional sibling module, when the tree has it
    from .backend_lab import *  # noqa: F403
