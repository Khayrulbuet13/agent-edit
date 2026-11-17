"""Build xview-ships-10 from the xView training download.

    python build_xview_ships10.py XVIEW_TRAIN_GEOJSON TILE_DIR OUT_DIR

XVIEW_TRAIN_GEOJSON  xView_train.geojson
TILE_DIR             the folder with the training tiles (*.tif)
OUT_DIR              receives images/ (one JPEG chip per labelled vessel) and
                     annotations_boxmask.json (COCO boxes, each also written as a rectangle polygon)

Ten vessel types. One chip per vessel: a window 3.5x the box (at least 160 px a side), long side
resized to 448-640 px. Every box of the ten types with at least half its area inside the window is
labelled. Per type, RandomState(2026).permutation: the first 12 go to eval, the next 88 at most to
the train pool. A train chip that shares a tile with any eval chip is dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 2026
N_EVAL_PER_CLASS = 12
MAX_PER_CLASS = 100
CONTEXT_SCALE = 3.5
MIN_CROP_SIDE = 160
LONG_SIDE_LO, LONG_SIDE_HI = 448, 640
OVERLAP_KEEP = 0.5
# The geojson lists some features twice, once per neighbouring tile, and the second copy carries
# the box of the other tile. Those overshoot their own tile by far, so they are dropped, not clipped.
OOB_TOLERANCE = 2.0

VESSEL_NAMES = {
    40: "maritime vessel", 41: "motorboat", 42: "sailboat", 44: "tugboat",
    45: "barge", 47: "fishing vessel", 49: "ferry", 50: "yacht",
    51: "container ship", 52: "oil tanker",
}
CLASS_NAMES = sorted(VESSEL_NAMES.values())
name_to_cid = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}


def load_instances(geojson: Path, tiles: Path) -> tuple[dict, dict]:
    """-> (by_class {type_id: [(tile, box, feature_id)]}, by_tile {tile: [(type_id, box, feature_id)]})"""
    data = json.loads(geojson.read_text())
    raw = []
    for feat in data["features"]:
        p = feat["properties"]
        tid = p.get("type_id")
        if tid not in VESSEL_NAMES:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in p.get("bounds_imcoords", "").split(","))
        except ValueError:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        raw.append((tid, p["image_id"], (x1, y1, x2, y2), p["feature_id"]))

    names = sorted({r[1] for r in raw})
    dims = {t: Image.open(tiles / t).size for t in names if (tiles / t).exists()}

    by_class: dict[int, list] = {t: [] for t in VESSEL_NAMES}
    by_tile: dict[str, list] = {}
    n_dropped = 0
    for tid, tile, (x1, y1, x2, y2), fid in raw:
        wh = dims.get(tile)
        if wh is None:
            continue
        W, H = wh
        if max(0.0, x2 - W, y2 - H, -x1, -y1) > OOB_TOLERANCE:
            n_dropped += 1
            continue
        box = (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))
        by_class[tid].append((tile, box, fid))
        by_tile.setdefault(tile, []).append((tid, box, fid))
    print(f"dropped {n_dropped} boxes that overshoot their tile by more than {OOB_TOLERANCE} px")
    return by_class, by_tile


def pick_split(by_class: dict) -> tuple[dict, dict, set]:
    """-> (eval rows, train candidate rows, tiles that hold an eval chip), each keyed by type_id."""
    eval_rows, train_cand, eval_tiles = {}, {}, set()
    for tid, rows in by_class.items():
        rows = sorted(rows, key=lambda r: (r[0], r[2]))
        n = len(rows)
        perm = np.random.RandomState(SEED).permutation(n)
        n_eval = min(N_EVAL_PER_CLASS, n)
        n_train = min(MAX_PER_CLASS, n) - n_eval
        eval_rows[tid] = [rows[i] for i in perm[:n_eval]]
        train_cand[tid] = [rows[i] for i in perm[n_eval:n_eval + max(0, n_train)]]
        eval_tiles.update(r[0] for r in eval_rows[tid])
    return eval_rows, train_cand, eval_tiles


def build_chip(tiles: Path, tile: str, bbox: tuple, by_tile: dict):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    cw, ch = max(w * CONTEXT_SCALE, MIN_CROP_SIDE), max(h * CONTEXT_SCALE, MIN_CROP_SIDE)
    img_path = tiles / tile
    if not img_path.exists():
        return None
    img = Image.open(img_path)
    W, H = img.size
    cx0, cy0 = max(0, cx - cw / 2), max(0, cy - ch / 2)
    cx1, cy1 = min(W, cx + cw / 2), min(H, cy + ch / 2)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    crop = img.crop((cx0, cy0, cx1, cy1)).convert("RGB")

    boxes = []
    for tid, (bx1, by1, bx2, by2), _fid in by_tile.get(tile, []):
        ix0, iy0 = max(bx1, cx0), max(by1, cy0)
        ix1, iy1 = min(bx2, cx1), min(by2, cy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        box_area = (bx2 - bx1) * (by2 - by1)
        if box_area <= 0 or (ix1 - ix0) * (iy1 - iy0) / box_area < OVERLAP_KEEP:
            continue
        boxes.append((tid, (ix0 - cx0, iy0 - cy0, ix1 - cx0, iy1 - cy0)))

    long_side = max(crop.size)
    scale = 1.0
    if long_side < LONG_SIDE_LO:
        scale = LONG_SIDE_LO / long_side
    elif long_side > LONG_SIDE_HI:
        scale = LONG_SIDE_HI / long_side
    if scale != 1.0:
        crop = crop.resize((max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))), Image.BICUBIC)
        boxes = [(tid, tuple(v * scale for v in b)) for tid, b in boxes]
    # Rebuild from raw bytes so the TIFF metadata does not reach the JPEG writer.
    crop = Image.frombytes("RGB", crop.size, crop.tobytes())
    return crop, boxes


def main(geojson: Path, tiles: Path, out_root: Path) -> None:
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    by_class, by_tile = load_instances(geojson, tiles)
    eval_rows, train_cand, eval_tiles = pick_split(by_class)

    images_coco, anns_coco = [], []
    keys = {"eval": [], "train": {name: [] for name in CLASS_NAMES}}
    n_same_tile = n_fail = 0

    def add_chip(tile: str, bbox: tuple, feature_id: int) -> str | None:
        nonlocal n_fail
        built = build_chip(tiles, tile, bbox, by_tile)
        if built is None:
            n_fail += 1
            return None
        crop, boxes = built
        key = f"chip_{Path(tile).stem}_{feature_id}"
        crop.save(images_out / f"{key}.jpg", quality=92)
        img_id = len(images_coco) + 1
        images_coco.append({"id": img_id, "file_name": f"images/{key}.jpg",
                            "width": crop.size[0], "height": crop.size[1]})
        for tid, (bx0, by0, bx1, by1) in boxes:
            bw, bh = bx1 - bx0, by1 - by0
            if bw <= 0 or bh <= 0:
                continue
            anns_coco.append({
                "id": len(anns_coco) + 1, "image_id": img_id, "category_id": name_to_cid[VESSEL_NAMES[tid]],
                "bbox": [bx0, by0, bw, bh], "area": bw * bh, "iscrowd": 0,
                "segmentation": [[bx0, by0, bx1, by0, bx1, by1, bx0, by1]],
            })
        return key

    for tid, name in VESSEL_NAMES.items():
        for tile, bbox, fid in eval_rows[tid]:
            key = add_chip(tile, bbox, fid)
            if key:
                keys["eval"].append(key)
        for tile, bbox, fid in train_cand[tid]:
            if tile in eval_tiles:
                n_same_tile += 1
                continue
            key = add_chip(tile, bbox, fid)
            if key:
                keys["train"][name].append(key)

    print(f"train candidates dropped (tile shared with an eval chip): {n_same_tile}")
    print(f"chips that failed to build: {n_fail}")
    print(f"eval chips: {len(keys['eval'])}  train chips: {sum(map(len, keys['train'].values()))}")
    coco = {"images": images_coco, "annotations": anns_coco,
            "categories": [{"id": name_to_cid[n], "name": n} for n in CLASS_NAMES]}
    (out_root / "annotations_boxmask.json").write_text(json.dumps(coco))
    print(f"wrote {len(images_coco)} images and {len(anns_coco)} annotations to {out_root}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
