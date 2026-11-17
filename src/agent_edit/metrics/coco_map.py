"""COCO-style mask AP over prediction docs (pycocotools COCOeval, segm)."""

from __future__ import annotations

from .match import gt_by_class


def mask_ap(recs, docs, cids) -> float:
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns = [], []
    for i, rec in enumerate(recs):
        h, w = rec["hw"]
        images.append({"id": i, "height": h, "width": w})
        for cid, rles in gt_by_class(rec).items():
            for r in rles:
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid, "segmentation": r,
                             "area": float(mu.area(r)), "bbox": mu.toBbox(r).tolist(), "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in cids]}
    gt.createIndex()
    key_to_id = {rec["image_key"]: i for i, rec in enumerate(recs)}
    dets = [{"image_id": key_to_id[k], "category_id": d["class_id"], "segmentation": d["rle"], "score": d["score"]}
            for k, doc in docs.items() for d in doc["instances"]]
    if not dets:
        return 0.0
    ev = COCOeval(gt, gt.loadRes(dets), "segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])
