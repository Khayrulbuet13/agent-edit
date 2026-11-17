"""Tiles and RLE geometry for PMT mask sources that cut big images into tiles.

A tiled cache entry is {"hw", "tile", "stride", "tiles": [entry, ...]}. Tiles of big images keep
their masks in tile coordinates ("local"): one full-size canvas per mask would not fit in memory.
"""

from __future__ import annotations

import numpy as np
import torch
from pycocotools import mask as mask_utils

from ... import config
from ...metrics.match import area


def tiles(hw, tile: int, stride: int) -> list[tuple]:
    """(y0, x0, y1, x1) windows covering the image, last row/col flush with the border."""
    def starts(n):
        if n <= tile:
            return [0]
        out = list(range(0, n - tile, stride))
        return out + [n - tile]
    return [(y, x, min(y + tile, hw[0]), min(x + tile, hw[1])) for y in starts(hw[0]) for x in starts(hw[1])]


def flat(entry: dict) -> dict:
    """A tiled entry as one bag of queries (tile j's query q -> index j * 200 + q).
    `win` holds the tile window of masks kept in tile coordinates, else None."""
    if "tiles" not in entry:
        return entry
    t = entry["tiles"]
    return {"q": torch.cat([x["q"] for x in t]), "mscore": torch.cat([x["mscore"] for x in t]),
            "rles": [r for x in t for r in x["rles"]], "hw": entry["hw"],
            "win": [x["window"] if x.get("local") else None for x in t for _ in x["rles"]]}


def paste(rle: dict, window, hw) -> dict:
    """Tile mask -> full-image RLE, built from the tile's runs (a dense canvas of a large tiled image is tens of Mpx)."""
    H, W = int(hw[0]), int(hw[1])
    y0, x0, y1, x1 = window
    m = mask_utils.decode(rle)
    t = np.concatenate([[0], m.T.reshape(-1).astype(np.int8), [0]])
    d = np.diff(t)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # runs in tile column-major order
    h = y1 - y0
    col_s, col_e = s // h, (e - 1) // h
    # a run never crosses a tile column unless the tile spans the full height; split those
    starts, ends = [], []
    for a, b, ca, cb in zip(s, e, col_s, col_e):
        for c in range(ca, cb + 1):
            lo, hi = max(a, c * h), min(b, (c + 1) * h)
            starts.append((x0 + c) * H + y0 + (lo - c * h))
            ends.append((x0 + c) * H + y0 + (hi - c * h))
    counts, pos = [], 0
    for a, b in zip(starts, ends):
        if counts and a == pos:  # touches the previous run in the canvas: merge
            counts[-1] += b - a
        else:
            counts += [a - pos, b - a]
        pos = b
    if H * W - pos:  # pycocotools leaves out an empty last run
        counts.append(H * W - pos)
    r = mask_utils.frPyObjects({"counts": [int(c) for c in counts], "size": [H, W]}, H, W)
    r["counts"] = r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]
    return r


def box_xywh(e: dict, i: int) -> list:
    x, y, bw, bh = mask_utils.toBbox(e["rles"][i]).tolist()
    w = e.get("win", [None] * len(e["rles"]))[i]
    return [x, y, bw, bh] if w is None else [x + w[1], y + w[0], bw, bh]


def iou_with_gt(e: dict, gt: list, idx=None) -> tuple[np.ndarray, np.ndarray]:
    """(IoU [M, G], share of each true instance covered [M, G]) for queries `idx` of a flat entry.
    Tile-local masks are compared with the GT cropped to their window; the union uses full areas."""
    idx = list(range(len(e["rles"]))) if idx is None else list(idx)
    M, G = len(idx), len(gt)
    iou, cov = np.zeros((M, G)), np.zeros((M, G))
    if not M or not G:
        return iou, cov
    ag = area(gt)
    win = e.get("win") or [None] * len(e["rles"])
    groups = {}
    for r, i in enumerate(idx):
        groups.setdefault(None if win[i] is None else tuple(win[i]), []).append(r)
    gb = mask_utils.toBbox(gt).reshape(-1, 4)
    for w, rows in groups.items():
        dt = [e["rles"][idx[r]] for r in rows]
        ad = area(dt)
        if w is None:
            cols, ref = list(range(G)), gt
        else:
            y0, x0, y1, x1 = w
            cols = [j for j in range(G) if gb[j, 0] < x1 and gb[j, 0] + gb[j, 2] > x0
                    and gb[j, 1] < y1 and gb[j, 1] + gb[j, 3] > y0]
            if not cols:
                continue
            ref = [mask_utils.encode(np.asfortranarray(mask_utils.decode(gt[j])[y0:y1, x0:x1])) for j in cols]
        ar = area(ref)
        u = mask_utils.iou(dt, ref, [0] * len(ref))
        inter = u * (ad[:, None] + ar[None]) / (1 + u)  # from u = I / (A + B - I)
        full = inter / np.maximum(ad[:, None] + ag[cols][None] - inter, 1)
        iou[np.ix_(rows, cols)] = full
        cov[np.ix_(rows, cols)] = inter / np.maximum(ag[cols][None], 1)
    return iou, cov


def dedupe_keep(rles, labels, order, dedupe: float = 0.5, cover_thr: float = 0.8) -> list:
    """Greedy cross-tile duplicate filter in score order: drop a prediction when a kept one of the same
    class has IoU >= dedupe with it or covers >= cover_thr of the smaller mask. Only kept masks whose
    boxes overlap are compared (others have IoU 0), so memory stays linear in the number of masks."""
    if not rles:
        return []
    a = area(rles)
    box = mask_utils.toBbox(rles).reshape(-1, 4)
    x0, y0, x1, y1 = box[:, 0], box[:, 1], box[:, 0] + box[:, 2], box[:, 1] + box[:, 3]
    kept = {}  # label -> kept indices
    out = []
    for i in order:
        ks = kept.setdefault(labels[i], [])
        if ks:
            j = np.asarray(ks)
            near = j[(x0[j] < x1[i]) & (x1[j] > x0[i]) & (y0[j] < y1[i]) & (y1[j] > y0[i])]
            if len(near):
                u = mask_utils.iou([rles[i]], [rles[t] for t in near], [0] * len(near))[0]
                inter = u * (a[i] + a[near]) / (1 + u)
                cov = inter / np.maximum(np.minimum(a[i], a[near]), 1)
                if (u >= dedupe).any() or (cov >= cover_thr).any():
                    continue
        ks.append(i)
        out.append(i)
    return out


if config.has_module("backends.pmt.tiling_lab"):  # names of an optional sibling module, when the tree has it
    from .tiling_lab import *  # noqa: F403
