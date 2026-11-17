"""Instance macro-F1 at IoU 0.5.

Per class, predictions are matched to true masks with `match.greedy_match`; TP/FP/FN are pooled
over images; F1 is averaged over classes that have any true instance or prediction.
"""

from __future__ import annotations

import numpy as np

from .docs import docs_by_key
from .match import IOU_THR, greedy_match, prf

METRIC = "instance_macro_f1_0.5"


def class_scores(records, docs, min_score=0.0, variant=None, iou=IOU_THR):
    """-> (macro F1, {class_id: {class_name, f1, precision, recall, tp, fp, fn, n_gt}})."""
    class_ids = sorted({int(c) for r in records for c in r["class_ids"]})
    names = records[0]["class_names"] if records else {}
    stats = {c: {"tp": 0, "fp": 0, "n_gt": 0} for c in class_ids}
    for rec in records:
        doc = docs.get(rec["image_key"])
        inst = []
        for d in (doc or {}).get("instances", []):
            if float(d["score"]) < min_score:
                continue
            if variant is not None and variant not in d.get("variants", []):
                continue
            inst.append((int(d["class_id"]), d["rle"], float(d["score"])))
        for cid in class_ids:
            gtm = list(rec["gt_by_cid"].get(cid, []))
            items = [(m, s) for c, m, s in inst if c == cid]
            _t, _f, _n, dets, _i = greedy_match([m for m, _s in items], [s for _m, s in items], gtm, iou)
            tp = sum(1 for _s, ok in dets if ok)
            st = stats[cid]
            st["tp"] += tp
            st["fp"] += len(dets) - tp
            st["n_gt"] += len(gtm)

    per_class, f1s = {}, []
    for cid in class_ids:
        st = stats[cid]
        tp, fp = st["tp"], st["fp"]
        fn = st["n_gt"] - tp
        p, r, f1 = prf(tp, fp, fn)
        per_class[cid] = {"class_name": names.get(cid, str(cid)), "f1": float(f1), "precision": float(p),
                          "recall": float(r), "tp": tp, "fp": fp, "fn": fn, "n_gt": st["n_gt"]}
        if tp + fp + fn > 0:
            f1s.append(f1)
    return (float(np.mean(f1s)) if f1s else 0.0), per_class


def score_records(project, records, preds, split, min_score=0.0, variant=None) -> dict:
    docs = docs_by_key(preds, [r["image_key"] for r in records])
    f1, per_class = class_scores(records, docs, float(min_score), variant)
    return {
        "project": project,
        "split": split,
        METRIC: f1,
        "per_class": per_class,
        "n_images": len(records),
        "n_missing_preds": len(records) - len(docs),
        "min_score": float(min_score),
    }


def score(project, preds, split="eval", n=None, seed=0, min_score=0.0, variant=None) -> dict:
    """Score `preds` on the project's eval list (or `split`).

    `min_score` drops predictions below that score before matching.
    `variant` keeps only instances whose `variants` list contains it; None keeps all.
    """
    from .. import data

    return score_records(project, data.load_split(project, split, n=n, seed=seed), preds, split, min_score, variant)


def score_train(project, preds, n=None, seed=0, min_score=0.0) -> dict:
    """Same matcher on the N training images: the agent's feedback signal."""
    from .. import data

    n = data.N_TRAIN if n is None else n
    return score_records(project, data.load_split(project, "train", n=n, seed=seed), preds, "train", min_score)


def counts(records, docs, min_score, iou=IOU_THR):
    """(n_img, n_cls, 3) [tp, fp, n_gt] per image and class, plus class ids. Same rules as class_scores."""
    cids = sorted({int(c) for r in records for c in r["class_ids"]})
    C = np.zeros((len(records), len(cids), 3), np.int64)
    for a, rec in enumerate(records):
        inst = [(int(d["class_id"]), d["rle"], float(d["score"]))
                for d in docs.get(rec["image_key"], {}).get("instances", []) if float(d["score"]) >= min_score]
        for b, cid in enumerate(cids):
            gt = list(rec["gt_by_cid"].get(cid, []))
            items = [(m, s) for c, m, s in inst if c == cid]
            _t, _f, _n, dets, _i = greedy_match([m for m, _ in items], [s for _, s in items], gt, iou)
            tp = sum(1 for _s, ok in dets if ok)
            C[a, b] = (tp, len(dets) - tp, len(gt))
    return C, cids


def f1_from_counts(C) -> tuple[float, np.ndarray]:
    """Macro-F1 over classes with any GT or prediction; C is (n_img, n_cls, 3) or (n_cls, 3)."""
    s = C.reshape(-1, C.shape[-2], 3).sum(0) if C.ndim == 3 else C
    tp, fp, ng = s[:, 0], s[:, 1], s[:, 2]
    fn = ng - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    used = (tp + fp + fn) > 0
    return (float(f1[used].mean()) if used.any() else 0.0), f1


def per_image_f1(C) -> np.ndarray:
    """Per image: mean F1 over classes with a GT or prediction there; 1.0 if none."""
    tp, fp, ng = C[..., 0], C[..., 1], C[..., 2]
    den = 2 * tp + fp + (ng - tp)
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    used = den > 0
    return np.where(used.any(1), (f * used).sum(1) / np.maximum(used.sum(1), 1), 1.0)


def summarize(project, records, docs, thr, split="eval") -> dict:
    """Pooled counts + official scorer, cross-checked; `_counts` feeds the bootstrap."""
    C, cids = counts(records, docs, thr)
    mf1, _ = f1_from_counts(C)
    official = score(project, docs, split=split, min_score=thr) if split == "eval" else score_train(project, docs, min_score=thr)
    off = official[METRIC]
    if abs(off - mf1) > 1e-9:
        raise RuntimeError(f"own counts {mf1} != official scorer {off}")
    pc = official["per_class"]
    return {
        "macro_f1": mf1,
        "per_class": {pc[c]["class_name"]: {k: pc[c][k] for k in ("f1", "precision", "recall", "tp", "fp", "fn", "n_gt")}
                      for c in cids},
        "n_images": len(records),
        "n_pred": int(sum(1 for d in docs.values() for x in d["instances"] if x["score"] >= thr)),
        "per_image_f1": [round(float(x), 6) for x in per_image_f1(C)],
        "_counts": C,
    }


def image_stats(rec, doc, thr, iou=IOU_THR) -> dict:
    """One image: F1 = 2TP / (2TP + FP + FN) per class with a GT or a prediction, averaged;
    1.0 if there is neither."""
    class_ids = sorted(int(c) for c in rec["class_ids"])
    by_cid = {c: [] for c in class_ids}
    for d in doc["instances"]:
        if float(d["score"]) >= thr:
            by_cid[int(d["class_id"])].append(d)
    f1s, n_pred, n_gt = [], {}, 0
    for cid in class_ids:
        gt = list(rec["gt_by_cid"].get(cid, []))
        items = by_cid[cid]
        tp, fp, fn, _, _ = greedy_match([d["rle"] for d in items], [d["score"] for d in items], gt, iou)
        n_pred[cid] = len(items)
        n_gt += len(gt)
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"f1": float(np.mean(f1s)) if f1s else 1.0, "n_pred": n_pred, "n_gt": n_gt}


def train_report(project, recs, docs, n_requested, seed, thr) -> dict:
    """The body of a `generate` answer: official train score plus per-image and per-class detail.
    Scored on exactly `recs`: re-loading the split by n would, for projects whose N counts images
    per class, load n images of every class and count the unscored ones as misses."""
    official = score_records(project, recs, docs, "train", thr)
    stats = [image_stats(r, docs[r["image_key"]], thr) for r in recs]
    pc = official["per_class"]
    counted = {k: v for k, v in pc.items() if v["tp"] + v["fp"] + v["fn"] > 0}
    n_pred_by_class = {k: 0 for k in pc}
    for s in stats:
        for k, v in s["n_pred"].items():
            n_pred_by_class[k] += v
    return {
        "macro_f1": official[METRIC],
        "per_image_f1": [round(s["f1"], 4) for s in stats],
        "n_images": len(recs),
        "n_images_requested": int(n_requested),
        "n_pred": int(sum(sum(s["n_pred"].values()) for s in stats)),
        "n_gt": int(sum(v["n_gt"] for v in pc.values())),
        "n_pred_per_image": [int(sum(s["n_pred"].values())) for s in stats],
        "n_gt_per_image": [int(s["n_gt"]) for s in stats],
        "per_class_f1": {v["class_name"]: round(v["f1"], 6) for v in counted.values()},
        "per_class": {
            v["class_name"]: {
                "f1": round(v["f1"], 6), "precision": round(v["precision"], 6),
                "recall": round(v["recall"], 6), "tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
                "n_gt": v["n_gt"], "n_pred": n_pred_by_class[k],
            }
            for k, v in pc.items()
        },
        "score_threshold": thr,
    }
