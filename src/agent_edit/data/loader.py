"""Loader for the manifest-based sets: COCO json annotations plus a folder of images."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # large tiles trip the decompression-bomb guard

SPLITS = ("train_pool", "eval")
_UNSUPPORTED = ("image_dirs", "extra_annotations", "cell_type_prefixes")


@cache
def _hw(path: str) -> tuple:
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


def image_hw(path) -> tuple:
    """(h, w) from the file header, in stored orientation (the EXIF turn is not applied)."""
    return _hw(str(path))


def open_rgb(path) -> np.ndarray:
    """H x W x 3 uint8 in stored orientation.

    The EXIF turn stays off: cached features and annotations are in stored orientation.
    """
    with Image.open(path) as im:
        if im.mode in ("RGBA", "P", "CMYK"):
            im = im.convert("RGB")
        arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@cache
def coco(annotations):
    from pycocotools.coco import COCO

    return COCO(str(Path(annotations).resolve()))


class CocoSource:
    """One annotation file and its image folder. An image key is a file stem, file name or COCO id."""

    def __init__(self, images, annotations):
        self.root = Path(images)
        self.annotations = Path(annotations)
        self.coco = coco(self.annotations)
        self._index = {}
        for img in self.coco.dataset["images"]:
            name = img["file_name"]
            for k in (Path(name).stem, name, str(img["id"])):
                self._index[k] = img  # a later image wins a shared key

    def image(self, key: str) -> dict:
        img = self._index.get(key)
        if img is None:
            raise KeyError(f"image_key {key!r} not in {self.annotations}")
        return img

    def path_of(self, key: str) -> Path:
        name = self.image(key)["file_name"]
        for p in (self.root / name, self.root / Path(name).name):  # some sets keep a flat folder
            if p.is_file():
                return p
        raise FileNotFoundError(f"{name} not under {self.root}")

    def check(self, key: str) -> Path:
        """Path of the image, after checking that the annotation canvas is the image size."""
        img, path = self.image(key), self.path_of(key)
        ann_hw = (int(img["height"]), int(img["width"]))
        if ann_hw != image_hw(path):
            raise ValueError(f"{key}: annotation canvas {ann_hw} differs from image {image_hw(path)}; "
                             "the masks would not line up")
        return path

    def gt(self, key: str, taxonomy: list) -> dict:
        """{class_id: [RLE]} in annotation order, every class present. Crowd boxes are kept."""
        to_ours = {}
        for t in taxonomy:
            for sid in t.get("source_cat_ids", [t["id"]]):
                to_ours.setdefault(int(sid), []).append(int(t["id"]))
        out = {int(t["id"]): [] for t in taxonomy}
        coco = self.coco
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[self.image(key)["id"]])):
            cids = to_ours.get(int(ann["category_id"]))
            if not cids:
                continue
            rle = coco.annToRLE(ann)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
            rle = {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
            for cid in cids:
                out[cid].append(rle)
        return out


@cache
def _source(images: str, annotations: str) -> CocoSource:
    return CocoSource(images, annotations)


def source(man: dict) -> CocoSource:
    if man.get("loader", "coco") != "coco" or any(k in man for k in _UNSUPPORTED):
        raise ValueError(f"manifest {man.get('project')!r} needs more than a plain COCO file and folder")
    return _source(str(man["source"]["images"]), str(man["source"]["annotations"]))


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def pool_keys(man: dict, split: str) -> tuple:
    _check_split(split)
    src = source(man)
    for key in man[split]:
        src.check(key)
    return tuple(man[split])


def image_paths(man: dict, keys) -> dict:
    src = source(man)
    return {k: str(src.path_of(k)) for k in keys}


def records(project: str, man: dict, split: str, keys: list) -> list:
    """One record per key, in key order. GT is RLE at the annotation canvas."""
    _check_split(split)
    src = source(man)
    names = {int(t["id"]): str(t["name"]) for t in man["taxonomy"]}
    out = []
    for key in keys:
        path = src.check(key)
        out.append({
            "image_key": str(key),
            "split": split,
            "path": str(path),
            "gt_by_cid": src.gt(key, man["taxonomy"]),
            "class_ids": list(names),
            "hw": list(image_hw(path)),
            "project": project,
            "class_names": names,
        })
    return out
