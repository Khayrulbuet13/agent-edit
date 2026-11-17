"""Prediction docs, one per image (the bridge's prediction-dump schema):

    {project, model, image_key, split, hw, instances: [{class_id, class_name, score, box, rle}]}

rle is {size: [h, w], counts: str}. An image with no doc counts as zero predictions.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..config import PKG


def key_file(key: str) -> str:
    return key.replace("/", "__")


def read_doc(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def docs_by_key(preds, keys) -> dict:
    """Accept {image_key: doc}, a list of docs, or a dir of per-image .json/.json.gz."""
    if isinstance(preds, dict):
        return {k: preds[k] for k in keys if k in preds}
    if isinstance(preds, (list, tuple)):
        want = set(keys)
        return {d["image_key"]: d for d in preds if d["image_key"] in want}
    root = Path(preds)
    out = {}
    for k in keys:
        for ext in (".json.gz", ".json"):
            p = root / f"{key_file(k)}{ext}"
            if p.is_file():
                out[k] = read_doc(p)
                break
    return out


def encode_masks(masks) -> list[dict]:
    """Bool masks (N, H, W) -> RLE dicts in the doc schema."""
    from pycocotools import mask as coco_mask

    out = []
    for m in masks:
        rle = coco_mask.encode(np.asfortranarray(np.asarray(m).astype(np.uint8)))
        out.append({"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"].decode()})
    return out


def make_instances(class_id, class_name, scores, boxes, masks) -> list[dict]:
    rles = encode_masks(masks)
    return [
        {"class_id": int(class_id), "class_name": class_name, "score": float(s), "box": [float(v) for v in b], "rle": r}
        for s, b, r in zip(scores, boxes, rles)
    ]


def make_doc(project, image_key, hw, instances, split="eval", model=PKG) -> dict:
    return {
        "project": project,
        "model": model,
        "image_key": image_key,
        "split": split,
        "hw": [int(hw[0]), int(hw[1])],
        "instances": list(instances),
    }


def write_pred_file(out_dir, project, image_key, hw, instances, split="eval", model=PKG) -> Path:
    """Write one image's doc as <out_dir>/<key>.json.gz (atomic)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{key_file(image_key)}.json.gz"
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(make_doc(project, image_key, hw, instances, split, model), fh)
    tmp.replace(dest)
    return dest
