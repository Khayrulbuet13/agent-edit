"""PMT cache for a frozen mask path: only the class head (rows + no-object) is editable.

Per image we keep the 200 final query embeddings (class_head input), each query's final mask as
RLE at image size, and its mask score, under $AGENT_EDIT_PMT_ROOT/cache/<project>[__<source>]/<key>.pt.

    agent-edit cache pmt PROJECT [PROJECT ...] [--shard i/n --splits a,b]

Tiled and SAM-mask (AMG) sources are built by lab_sets.py, when it is present.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from ... import config, data
from . import head as H

VERSION = 1


def cache_root() -> Path:
    return config.pmt_root() / "cache"


def cache_project(project: str) -> str:
    """The project folder a cache actually lives under: `project` itself, unless its manifest
    names a parent to read from instead (manifest field `cache_from`, e.g. a mix5 set that only
    regroups another set's own classes over the same images -- no re-encode needed)."""
    return data.manifest(project).get("cache_from") or project


def source_root(project: str, source: str = "pmt") -> Path:
    """'pmt' = one pass per image; 'pmt_tile256' etc. = tiled passes; 'amg_<variant>' = SAM masks."""
    base = cache_project(project)
    return cache_root() / (base if source == "pmt" else f"{base}__{source.removeprefix('pmt_')}")


def encode(dec, weights: dict, feats: torch.Tensor, rope, hw, canvas=None, offset=(0, 0)) -> dict:
    """hw: size of the image PMT saw. canvas/offset: paste its masks into a bigger image (tiling)."""
    got = []
    hook = dec.class_head.register_forward_hook(lambda m, i, o: got.append(i[0]))
    try:
        with torch.no_grad():
            m, c = H.head_forward(dec, weights, feats, rope)
    finally:
        hook.remove()
    q = got[-1][0].float()
    hw = tuple(int(v) for v in hw)
    sh, sw = H.Backbone.scaled_size(hw)
    # interpolation is per query, so chunks give the same values; big aerial tiles need it
    step = max(1, min(H.NUM_Q, int(4e8 // (hw[0] * hw[1]))))
    mscores, rles = [], []
    for q0 in range(0, H.NUM_Q, step):
        with torch.no_grad():
            ml = F.interpolate(m[:, q0:q0 + step].float(), (H.RES, H.RES), mode="bilinear")
            ml = F.interpolate(ml[:, :, :sh, :sw], hw, mode="bilinear")[0]
            binm = ml > 0
            mscores.append((ml.sigmoid().flatten(1) * binm.flatten(1)).sum(1) / (binm.flatten(1).sum(1) + 1e-6))
        bm = binm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if canvas is not None:
            full = np.zeros((canvas[0], canvas[1], bm.shape[2]), np.uint8)
            full[offset[0]:offset[0] + hw[0], offset[1]:offset[1] + hw[1]] = bm
            bm = full
        rles += mask_utils.encode(np.asfortranarray(bm))
        del ml, binm
    mscore = torch.cat(mscores)
    for r in rles:
        r["counts"] = r["counts"].decode()
    return {"v": VERSION, "q": q.cpu(), "mscore": mscore.cpu(), "rles": rles,
            "hw": tuple(canvas) if canvas is not None else hw, "class_logits": c[0].float().cpu()}


def path(root: Path, key) -> Path:
    return Path(root) / f"{key}.pt"


def save(root: Path, key, entry: dict) -> None:
    p = path(root, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(entry, tmp)
    tmp.replace(p)


def load(root: Path, key) -> dict:
    return torch.load(path(root, key), weights_only=False)


def project_keys(project: str) -> dict:
    out = {"train_pool": list(data.pool_keys(project, "train_pool")), "eval": list(data.pool_keys(project, "eval"))}
    if project not in data.EXTRA_PROJECTS:
        out["train_extra"] = list(data.extra_pool(project))  # only N=100 runs read these
    return out


def _lab():
    if importlib.util.find_spec(".lab_sets", __package__) is None:
        raise RuntimeError("tiled and AMG mask sources need lab_sets.py, which this tree does not have")
    from . import lab_sets

    return lab_sets


def build(project: str, splits=("train_pool", "eval"), device="cuda", tile: int = 0, stride: int = 0,
          batch: int = 0, shard: int = 0, nshards: int = 1) -> None:
    """Cache every image of the given splits once; skips cached images, so it resumes."""
    lab = _lab() if tile else None
    # batched tiles resize masks straight to the tile, so they are their own source ("b")
    root = cache_root() / (f"{project}__tile{tile}{'b' if batch else ''}" if tile else project)
    keys = project_keys(project)
    # shard on the full list, so jobs started at different times still take disjoint images
    todo = [(s, k) for s in splits for k in keys[s]][shard::nshards]
    todo = [(s, k) for s, k in todo if not path(root, k).exists()]
    print(f"[pmt_cache] {project}: {sum(len(keys[s]) for s in splits)} images, {len(todo)} to cache", flush=True)
    if not todo:
        return
    bb, dec, w = H.Backbone(device), H.build_decoder(device), H.load_head_weights(device)
    t0 = time.time()
    for s in splits:
        ks = [k for s2, k in todo if s2 == s]
        # records() knows only train_pool/eval; extras come from the train_pool source
        for rec in data.records(project, "train_pool" if s == "train_extra" else s, ks):
            img = np.asarray(data.open_rgb(rec["path"]))
            if tile:
                e = lab.encode_tiled(bb, dec, w, img, tile, stride, batch)
            else:
                f, rope = bb.features(img)
                e = encode(dec, w, f, rope, img.shape[:2])
            e["split"] = s
            save(root, rec["image_key"], e)
    print(f"[pmt_cache] {project}: done in {time.time() - t0:.0f}s", flush=True)


if config.has_module("backends.pmt.cache_lab"):  # names of an optional sibling module, when the tree has it
    from .cache_lab import *  # noqa: F403
