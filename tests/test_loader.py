"""The own loader on a tiny COCO set built in tmp_path. No project data, no AGENT_EDIT_* variables."""

import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_util

from agent_edit import data
from agent_edit.data import loader, registry, splits

W, H = 32, 24
TAX = [  # class 2 comes first on purpose: records must follow taxonomy order, not id order
    {"id": 2, "name": "wide_(thing)", "source_cat_ids": [2, 3]},
    {"id": 1, "name": "one", "source_cat_ids": [1]},
]


def _poly(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _ann(i, img, cat, box, crowd=0):
    x0, y0, x1, y1 = box
    return {"id": i, "image_id": img, "category_id": cat, "iscrowd": crowd, "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": (x1 - x0) * (y1 - y0), "segmentation": _poly(*box)}


def _rgb(seed=0, w=W, h=H):
    return Image.fromarray(np.random.RandomState(seed).randint(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.fixture
def tiny(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    _rgb(1).save(root / "rgb.jpg")
    Image.fromarray(np.random.RandomState(2).randint(0, 255, (H, W), dtype=np.uint8)).save(root / "grey.jpg")
    _rgb(3).convert("RGBA").save(root / "rgba.png")
    _rgb(4).save(root / "png_as.jpg", format="PNG")
    exif = Image.Exif()
    exif[0x0112] = 6  # viewers turn it by 90 degrees
    _rgb(5, w=40, h=20).save(root / "exif.jpg", exif=exif)
    Image.fromarray(np.arange(W * H, dtype=np.uint16).reshape(H, W) * 7).save(root / "deep.png")
    _rgb(6).save(root / "flat.jpg")  # the annotation says train/flat.jpg
    images = [
        {"id": 10, "file_name": "rgb.jpg", "height": H, "width": W},
        {"id": 11, "file_name": "grey.jpg", "height": H, "width": W},
        {"id": 12, "file_name": "rgba.png", "height": H, "width": W},
        {"id": 13, "file_name": "png_as.jpg", "height": H, "width": W},
        {"id": 14, "file_name": "exif.jpg", "height": 20, "width": 40},
        {"id": 15, "file_name": "deep.png", "height": H, "width": W},
        {"id": 16, "file_name": "train/flat.jpg", "height": H, "width": W},
        {"id": 20, "file_name": "x/dup.jpg", "height": H, "width": W},
        {"id": 21, "file_name": "y/dup.jpg", "height": H, "width": W},
    ]
    anns = [
        _ann(1, 10, 1, (2, 2, 10, 10)),
        _ann(2, 10, 3, (4, 4, 20, 8)),
        _ann(3, 10, 2, (1, 1, 5, 5)),
        _ann(4, 10, 9, (0, 0, 3, 3)),  # source id outside the taxonomy
        _ann(5, 10, 1, (12, 12, 16, 16), crowd=1),
    ]
    for i in range(8):  # plain images for the split tests: a0..a3 hold class 1, b0..b3 class 2
        name = f"{'ab'[i // 4]}{i % 4}"
        _rgb(10 + i).save(root / f"{name}.jpg")
        images.append({"id": 100 + i, "file_name": f"{name}.jpg", "height": H, "width": W})
        anns.append(_ann(100 + i, 100 + i, 1 + i // 4, (1, 1, 9, 9)))
    cats = [{"id": c, "name": f"c{c}"} for c in (1, 2, 3, 9)]
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return root, ann_path


def _area(rle):
    return int(mask_util.area({"size": rle["size"], "counts": rle["counts"].encode()}))


def test_keys_stem_name_and_id_reach_one_image(tiny):
    src = loader.CocoSource(*tiny)
    assert src.image("rgb")["id"] == src.image("rgb.jpg")["id"] == src.image("10")["id"] == 10
    assert src.image("flat")["file_name"] == "train/flat.jpg"
    with pytest.raises(KeyError):
        src.image("nope")


def test_last_image_wins_a_shared_key(tiny):
    assert loader.CocoSource(*tiny).image("dup")["id"] == 21


def test_path_falls_back_to_basename(tiny):
    root, ann = tiny
    src = loader.CocoSource(root, ann)
    assert src.path_of("rgb") == root / "rgb.jpg"
    assert src.path_of("flat") == root / "flat.jpg"
    with pytest.raises(FileNotFoundError):
        src.path_of("dup")  # in the annotations only


def test_hw_is_stored_orientation(tiny):
    root, _ = tiny
    assert loader.image_hw(root / "rgb.jpg") == (H, W)
    assert loader.image_hw(root / "exif.jpg") == (20, 40)  # a turn would give (40, 20)
    assert data.open_rgb(root / "exif.jpg").shape == (20, 40, 3)


@pytest.mark.parametrize("name", ["rgb.jpg", "grey.jpg", "rgba.png", "png_as.jpg", "deep.png"])
def test_open_rgb_gives_hwx3_uint8(tiny, name):
    arr = data.open_rgb(tiny[0] / name)
    assert arr.shape == (H, W, 3) and arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]


def test_open_rgb_grey_stacks_one_channel_three_times(tiny):
    arr = data.open_rgb(tiny[0] / "grey.jpg")
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()


def test_open_rgb_rgba_drops_alpha_and_keeps_pixels(tiny):
    arr = data.open_rgb(tiny[0] / "rgba.png")
    assert np.array_equal(arr, np.asarray(_rgb(3)))


def test_open_rgb_stretches_deep_images_to_the_1_99_percentile(tiny):
    arr = data.open_rgb(tiny[0] / "deep.png")[..., 0]
    assert arr.min() == 0 and arr.max() == 255
    assert (np.diff(arr.ravel().astype(int)) >= 0).all()  # the ramp stays monotone


def test_gt_is_one_rle_list_per_class_in_annotation_order(tiny):
    gt = loader.CocoSource(*tiny).gt("rgb", TAX)
    assert list(gt) == [2, 1]
    assert [_area(r) for r in gt[2]] == [16 * 4, 4 * 4]  # source 3 then source 2: two ids, one class
    assert [_area(r) for r in gt[1]] == [8 * 8, 4 * 4]  # the crowd box stays
    assert all(r["size"] == [H, W] and isinstance(r["counts"], str) for r in gt[1] + gt[2])


def test_gt_has_every_class_even_when_empty(tiny):
    gt = loader.CocoSource(*tiny).gt("grey", TAX)
    assert gt == {2: [], 1: []}


def test_size_mismatch_is_an_error_not_a_silent_drop(tiny, tmp_path):
    root, ann = tiny
    doc = json.loads(ann.read_text())
    doc["images"][0]["height"] += 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="annotation canvas"):
        loader.CocoSource(root, bad).check("rgb")


@pytest.fixture
def project(tiny, tmp_path, monkeypatch):
    """A manifest named like a registered set, found through AGENT_EDIT_DATA."""
    root, ann = tiny
    man = {
        "project": "cub-10", "loader": "coco", "taxonomy": TAX,
        "source": {"images": str(root), "annotations": str(ann)},
        "train_pool": [f"{c}{i}" for c in "ab" for i in range(4)],
        "train_pool_by_class": {"a": [f"a{i}" for i in range(4)], "b": [f"b{i}" for i in range(4)]},
        "eval": ["rgb", "exif"],
    }
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "cub-10.json").write_text(json.dumps(man))
    monkeypatch.setenv("AGENT_EDIT_DATA", str(tmp_path))
    caches = (registry.manifest, splits.pool_keys, splits._load_split_cached, loader.coco, loader._source)
    for c in caches:
        c.cache_clear()
    yield "cub-10"
    for c in caches:
        c.cache_clear()


def test_record_shape_and_class_order(project):
    (rec,) = data.records(project, "eval", ["rgb"])
    assert list(rec) == ["image_key", "split", "path", "gt_by_cid", "class_ids", "hw", "project", "class_names"]
    assert rec["image_key"] == "rgb" and rec["split"] == "eval" and rec["project"] == project
    assert isinstance(rec["path"], str) and rec["path"].endswith("rgb.jpg")
    assert rec["hw"] == [H, W]
    assert rec["class_ids"] == [2, 1] and list(rec["gt_by_cid"]) == [2, 1]
    assert rec["class_names"] == {2: "wide_(thing)", 1: "one"}


def test_records_follow_the_key_order_given(project):
    recs = data.records(project, "eval", ["exif", "rgb"])
    assert [r["image_key"] for r in recs] == ["exif", "rgb"]
    assert recs[0]["hw"] == [20, 40]


def test_pool_keys_are_the_manifest_lists(project):
    assert data.pool_keys(project, "eval") == ("rgb", "exif")
    assert data.eval_image_keys(project) == ["rgb", "exif"]
    assert set(data.eval_image_paths(project)) == {"rgb", "exif"}
    assert len(data.pool_keys(project, "train_pool")) == 8


def test_train_keys_interleave_classes_in_the_seeded_order(project):
    # RandomState(1000 + seed).permutation(4) is [2 1 0 3] for seed 0 and [0 2 3 1] for seed 1.
    assert data.train_keys(project, 2, 0) == ["a2", "b2", "a1", "b1"]
    assert data.train_keys(project, 3, 1) == ["a0", "b0", "a2", "b2", "a3", "b3"]
    assert sorted(data.train_keys(project, 10, 0)) == sorted(data.pool_keys(project, "train_pool"))


def test_load_split_train_matches_train_keys(project):
    recs = data.load_split(project, "train", n=2, seed=0)
    assert [r["image_key"] for r in recs] == data.train_keys(project, 2, 0)
    assert all(r["split"] == "train_pool" for r in recs)


def test_agent_gt_has_no_eval_images(project):
    doc = data.gt_for_agent(project, n=1, seed=0)
    assert [i["image_key"] for i in doc["images"]] == ["a2", "b2"]


def test_a_missing_file_stops_the_pool_instead_of_shrinking_it(project, tiny):
    (tiny[0] / "a0.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        data.pool_keys(project, "train_pool")


def test_manifest_with_a_loader_this_code_lacks_is_refused():
    with pytest.raises(ValueError, match="plain COCO"):
        loader.source({"project": "x", "loader": "voc", "source": {"images": "i", "annotations": "a"}})


def test_bridge_sets_are_listed_only_with_the_bridge():
    assert set(data.BRIDGE_PROJECTS) <= set(data.PROJECTS)
    assert set(data.PROJECTS) - set(data.BRIDGE_PROJECTS) == set(data.EXTRA_PROJECTS)
