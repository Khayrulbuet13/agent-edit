"""Error pictures of train images for the agent (Look-then-Edit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics.match import IOU_THR, rle_of

OVERLAY_SIDE = 1024
COLORS = {"tp": (0, 200, 0), "missed": (230, 0, 0), "false": (255, 210, 0), "wrong_class": (40, 90, 255)}
LEGEND = ("outlines: green = correct, red = missed true instance, yellow = false mask, "
          "blue = right object but wrong class")
SHEET_K = 16  # worst-scored train images tiled into one sheet.png, --overlay-sheet
SHEET_COLS = 4
SHEET_CELL = 256


def draw_overlay(rgb: np.ndarray, rec, doc, thr, iou=IOU_THR) -> np.ndarray:
    """Outlines of correct (green), missed true (red), false (yellow) and wrong-class (blue) masks."""
    from pycocotools import mask as mask_utils
    from scipy import ndimage

    h, w = rgb.shape[:2]
    out = rgb.copy()
    inst = [d for d in doc["instances"] if d["score"] >= thr]
    gt_all = [(cid, rle_of(r)) for cid, rs in rec["gt_by_cid"].items() for r in rs]
    paint = []
    for cid in sorted(rec["gt_by_cid"]):
        items = sorted([d for d in inst if d["class_id"] == cid], key=lambda d: -d["score"])
        gt = [rle_of(r) for r in rec["gt_by_cid"][cid]]
        used = np.zeros(len(gt), bool)
        ious = mask_utils.iou([d["rle"] for d in items], gt, [0] * len(gt)) if items and gt else None
        for a, d in enumerate(items):  # the scorer's greedy rule: best unused true mask, in score order
            row = np.where(used, -1.0, ious[a]) if ious is not None else np.array([-1.0])
            j = int(np.argmax(row))
            if row[j] >= iou:
                used[j] = True
                paint.append((d["rle"], "tp"))
                continue
            other = [g for c, g in gt_all if c != cid]
            wrong = bool(other) and mask_utils.iou([d["rle"]], other, [0] * len(other)).max() >= iou
            paint.append((d["rle"], "wrong_class" if wrong else "false"))
        paint += [(g, "missed") for j, g in enumerate(gt) if not used[j]]
    for rle, kind in paint:
        m = mask_utils.decode(rle).astype(bool)
        edge = m & ~ndimage.binary_erosion(m, iterations=max(1, round(max(h, w) / 400)))
        out[edge] = COLORS[kind]
    s = OVERLAY_SIDE / max(h, w)
    if s < 1:
        from PIL import Image

        out = np.asarray(Image.fromarray(out).resize((round(w * s), round(h * s))))
    return out


def render_one(rec, doc, thr):
    """The annotated image array for one record (what draw_errors saves to PNG)."""
    from ..data import open_rgb

    return draw_overlay(np.asarray(open_rgb(rec["path"])), rec, doc, thr)


def draw_errors(recs, idx, docs, out_dir, thr) -> list[str]:
    """One error picture per image, named by its train index."""
    from PIL import Image

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for rec, i in zip(recs, idx):
        img = render_one(rec, docs[rec["image_key"]], thr)
        p = d / f"train_{i:03d}.png"
        Image.fromarray(img).save(p)
        files.append(str(p))
    return files


def worst_k(per_image_f1, k) -> list[int]:
    """Positions (into the caller's own image order) of the k worst-scored images, worst first."""
    order = sorted(range(len(per_image_f1)), key=lambda j: per_image_f1[j])
    return order[:k]


def draw_sheet(tiles, out_dir, name="sheet.png", cols=SHEET_COLS, cell=SHEET_CELL) -> str:
    """One PNG tiling `tiles` (caller already ordered worst first and capped at SHEET_K):
    [(annotated rgb array, one-line label), ...]. A quick look at the worst of a pass."""
    from PIL import Image, ImageDraw

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    n = max(1, len(tiles))
    cols = max(1, min(cols, n))
    rows = -(-n // cols)
    label_h = 16
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(tiles):
        im = img if isinstance(img, Image.Image) else Image.fromarray(img)
        im = im.resize((cell, cell))
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + label_h)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + cell + 1), label[:60], fill=(0, 0, 0))
    p = d / name
    sheet.save(p)
    return str(p)
