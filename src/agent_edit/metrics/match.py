"""Greedy score-ordered mask matcher on COCO RLEs.

Per class, predictions walk in descending score order (stable sort); each claims the unused
true instance with the highest IoU (first maximum on ties) and is a TP if that IoU >= the cut.
IoUs come from pycocotools on RLEs, so huge tiles are never decoded to dense masks.
"""

from __future__ import annotations

import numpy as np

IOU_THR = 0.5
# Cap on the IoU block held at once (float64 cells): 32 MB however many instances a tile has.
IOU_BLOCK_CELLS = 4_000_000


def rle_of(blob) -> dict:
    """COCO RLE (bytes counts) from an RLE dict or a dense mask. Never decodes."""
    from pycocotools import mask as mask_utils

    if isinstance(blob, dict) and "counts" in blob and "size" in blob:
        counts = blob["counts"]
        size = [int(blob["size"][0]), int(blob["size"][1])]
        if isinstance(counts, (list, tuple, np.ndarray)):
            return mask_utils.frPyObjects({"size": size, "counts": [int(v) for v in counts]}, size[0], size[1])
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        elif isinstance(counts, np.bytes_):
            counts = bytes(counts)
        return {"size": size, "counts": counts}
    m = np.asarray(blob)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {m.shape}")
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))


def rle_iou(a_rles, b_rles) -> np.ndarray:
    from pycocotools import mask as mask_utils

    sizes = {tuple(r["size"]) for r in a_rles} | {tuple(r["size"]) for r in b_rles}
    if len(sizes) > 1:
        # pycocotools would return a plausible wrong number for masks of different canvases
        raise ValueError(f"IoU across mismatched canvases: {sorted(sizes)}")
    return np.asarray(mask_utils.iou(list(a_rles), list(b_rles), [0] * len(b_rles)), np.float64)


def greedy_match(pred_blobs, pred_scores, gt_blobs, iou_thr: float = IOU_THR):
    """-> (tp, fp, fn, dets [(score, is_tp)] in claim order, matched IoUs)."""
    if len(pred_blobs) != len(pred_scores):
        raise ValueError(f"pred_masks/pred_scores length mismatch: {len(pred_blobs)} != {len(pred_scores)}")
    n_gt = len(gt_blobs)
    if not len(pred_blobs):
        return 0, 0, n_gt, [], []
    scores = np.asarray(pred_scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    if n_gt == 0:
        return 0, len(pred_blobs), 0, [(float(scores[pi]), False) for pi in order], []
    gt_r = [rle_of(b) for b in gt_blobs]
    used = np.zeros(n_gt, bool)
    tp = fp = 0
    dets, matched_ious = [], []
    step = max(1, IOU_BLOCK_CELLS // n_gt)
    for lo in range(0, len(order), step):
        idx = order[lo : lo + step]
        block = rle_iou([rle_of(pred_blobs[i]) for i in idx], gt_r)
        for r, pi in enumerate(idx):
            row = np.where(used, -1.0, block[r])
            j = int(np.argmax(row))
            best_iou = float(row[j])
            if best_iou <= 0.0:
                best_iou, j = 0.0, -1
            is_tp = j >= 0 and best_iou >= iou_thr
            if is_tp:
                used[j] = True
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1
            dets.append((float(scores[pi]), bool(is_tp)))
    return tp, fp, n_gt - tp, dets, matched_ious


def prf(tp, fp, fn) -> tuple[float, float, float]:
    """(precision, recall, F1); a zero denominator gives 0.0."""
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def gt_by_class(rec) -> dict:
    """{class_id: [compressed RLE, ...]} of a record; pycocotools area() needs the compressed form."""
    from pycocotools import mask as mask_utils

    out = {}
    for cid in sorted(rec["gt_by_cid"]):
        out[int(cid)] = []
        for r in rec["gt_by_cid"][cid]:
            r = rle_of(r)
            if isinstance(r["counts"], list):
                r = mask_utils.frPyObjects(r, *r["size"])
            out[int(cid)].append(r)
    return out


def area(rles) -> np.ndarray:
    from pycocotools import mask as mask_utils

    # pycocotools 2.0.11 area() overflows on lists longer than 255
    if not rles:
        return np.zeros(0)
    return np.concatenate([mask_utils.area(rles[i:i + 200]) for i in range(0, len(rles), 200)]).astype(float)
