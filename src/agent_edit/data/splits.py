"""Train and eval splits."""

from __future__ import annotations

import json
import os
import tempfile
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from . import loader
from .registry import (
    EXTRA_PROJECTS,
    N_TRAIN,
    PER_CLASS_SHOT,
    bridge,
    check_project,
    class_prompts,
    class_table,
    manifest,
)

EXTRA_SEED = 2026
N_EXTRA = 40  # N=100 = the 60-image pool + 40 more from the same source; eval never changes


def seed_order(n: int, seed: int) -> np.ndarray:
    """Which pool images a seed picks. The stream is fixed: changing it changes every earlier run's images."""
    return np.random.RandomState(1000 + int(seed)).permutation(int(n))


def _bridge_records(project: str, man: dict, split: str) -> list:
    """The bridge loader reads a manifest file, so hand it one with the resolved paths."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(man, f)
    try:
        return bridge().common().load_records(project, f.name, split=split, load_gt=False)
    finally:
        os.unlink(f.name)


@cache
def pool_keys(project: str, split: str) -> tuple:
    """Image keys of a split. Own sets fail on a missing file or a size mismatch; bridge sets drop them."""
    man = manifest(project)
    if project in EXTRA_PROJECTS:
        return loader.pool_keys(man, split)
    return tuple(r["image_key"] for r in _bridge_records(project, man, split))


@cache
def _extra_keys(project: str) -> tuple:
    """Source images outside train_pool and eval that hold a project class, in a fixed random order."""
    man = manifest(project)
    coco = loader.coco(man["source"]["annotations"])
    taken = set(man["train_pool"]) | set(man["eval"])
    tax = man["taxonomy"]
    if man.get("cell_type_prefixes"):
        names = {t["name"] for t in tax}

        def ok(img, anns):
            return Path(img["file_name"]).name.split("_")[0] in names and anns
    else:
        src = {s for t in tax for s in t.get("source_cat_ids", [t["id"]])}

        def ok(img, anns):
            return any(a["category_id"] in src for a in anns)

    keys = []
    for img in coco.dataset["images"]:
        key = Path(img["file_name"]).stem
        if key in taken:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img["id"]], iscrowd=False))
        if ok(img, anns):
            keys.append(key)
    keys = sorted(keys)
    order = np.random.RandomState(EXTRA_SEED).permutation(len(keys))
    return tuple(keys[i] for i in order[: 3 * N_EXTRA])  # spares for images the loader drops


@cache
def extra_pool(project: str) -> tuple:
    """The first N_EXTRA extra keys the bridge loader can actually read."""
    man = dict(manifest(project))
    man["train_pool"] = list(_extra_keys(project))
    return tuple(r["image_key"] for r in _bridge_records(project, man, "train_pool"))[:N_EXTRA]


def train_keys(project: str, n: int, seed: int) -> list[str]:
    if project in PER_CLASS_SHOT:
        kept = set(pool_keys(project, "train_pool"))
        cols = []
        for _name, keys in sorted(manifest(project)["train_pool_by_class"].items()):
            keys = [k for k in keys if k in kept]
            order = seed_order(len(keys), seed)
            cols.append([keys[i] for i in order[: min(int(n), len(keys))]])
        return [c[j] for j in range(max(map(len, cols))) for c in cols if j < len(c)]
    pool = pool_keys(project, "train_pool")
    order = seed_order(len(pool), seed)
    keys = [pool[i] for i in order[: min(int(n), len(pool))]]
    if int(n) > len(pool):
        keys += list(extra_pool(project))[: int(n) - len(pool)]
    return keys


def records(project: str, split: str, keys: list[str]) -> list[dict]:
    """Records for `keys` in the given order, GT as RLE."""
    if project in EXTRA_PROJECTS:
        return loader.records(project, manifest(project), split, keys)
    ds_loader = bridge().loader()
    normalize = bridge().common()._normalize_record
    man = dict(manifest(project))
    man[split] = list(keys)
    names = {c["class_id"]: c["class_name"] for c in class_table(project)}
    out = []
    for item in ds_loader.iter_images(man, split, load_pixels=False, load_gt=True):
        if item.get("canvas") or not Path(item["path"]).is_file():
            continue
        rec = normalize({"image_key": item["image_key"], "split": split, "path": item["path"],
                         "gt": item["gt"], "class_ids": list(names)})
        rec["hw"] = list(ds_loader.image_hw(rec["path"]))
        rec["project"] = project
        rec["class_names"] = names
        rec["path"] = str(rec["path"])
        rec.pop("rgb", None)
        rec.pop("missing", None)
        out.append(rec)
    by_key = {r["image_key"]: r for r in out}
    return [by_key[k] for k in keys if k in by_key]


@lru_cache(maxsize=16)
def _load_split_cached(project: str, split: str, n, seed: int) -> tuple:
    if split == "train":
        keys = train_keys(project, N_TRAIN if n is None else n, seed)
        return tuple(records(project, "train_pool", keys))
    if split == "eval":
        keys = list(pool_keys(project, "eval"))
        if n is not None:
            keys = keys[: int(n)]
        return tuple(records(project, "eval", keys))
    raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


def load_split(project: str, split: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Records: {image_key, path, hw, gt_by_cid {cid: [RLE]}, class_ids, class_names, split, project}.

    'train' is the first N of the seed's order over the 60-image train pool
    (RandomState(1000 + seed).permutation). 'eval' is the eval list as pool_keys returns it.
    Only scorers may read 'eval'.
    """
    check_project(project)
    return list(_load_split_cached(project, split, n, int(seed)))


def gt_for_agent(project: str, n: int = N_TRAIN, seed: int = 0) -> dict:
    """Train-split labels only. Eval GT is not reachable from here."""
    recs = load_split(project, "train", n=n, seed=seed)
    return {
        "project": project,
        "classes": class_table(project),
        "prompts": class_prompts(project),
        "images": [{k: r[k] for k in ("image_key", "path", "hw", "gt_by_cid")} for r in recs],
    }


def write_agent_gt(project: str, out_path, n: int = N_TRAIN, seed: int = 0) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = gt_for_agent(project, n=n, seed=seed)
    for im in doc["images"]:
        im["gt_by_cid"] = {
            str(c): [{"size": r["size"], "counts": r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]}
                     for r in rles]
            for c, rles in im["gt_by_cid"].items()
        }
    out.write_text(json.dumps(doc))
    return out


def eval_image_keys(project: str) -> list[str]:
    """Eval keys only, no GT."""
    return list(pool_keys(project, "eval"))


def eval_image_paths(project: str) -> dict:
    """{image_key: path}, no GT."""
    man = manifest(project)
    if project in EXTRA_PROJECTS:
        return loader.image_paths(man, eval_image_keys(project))
    ds_loader = bridge().loader()
    reader = ds_loader._reader_for(man, Path(man["source"]["images"]), Path(man["source"]["annotations"]))
    return {k: str(reader.path_of(k)) for k in eval_image_keys(project)}
