"""--overlay-sheet on the PMT backend: the sheet label names only the class a record has ground truth for.

gt_by_cid carries every project class as a key, most with an empty list, so the label must look at
the lists and not at the keys."""

import pytest

from conftest import need_manifest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _manifests():
    need_manifest("cub-10")


def _empty_rle():
    return {"size": [1, 1], "counts": "0"}


def test_sheet_label_lists_only_the_class_with_gt():
    from agent_edit.backends.pmt import backend as PB

    be = PB.make_backend("cub-10")
    present = be.class_ids[2]
    rec = {"gt_by_cid": {cid: ([_empty_rle()] if cid == present else []) for cid in be.class_ids}}
    doc = {"instances": []}

    label = be._sheet_label(7, rec, doc, 0.0)

    assert label.count("GT:") == 1
    gt_part = label.split("GT:", 1)[1].split(" pred:")[0]
    assert gt_part == be.names[present]


def test_sheet_label_names_the_top_scoring_prediction():
    from agent_edit.backends.pmt import backend as PB

    be = PB.make_backend("cub-10")
    present = be.class_ids[0]
    other = be.names[be.class_ids[1]]
    rec = {"gt_by_cid": {cid: ([_empty_rle()] if cid == present else []) for cid in be.class_ids}}
    doc = {"instances": [{"score": 0.6, "class_name": other}, {"score": 0.9, "class_name": other}]}

    label = be._sheet_label(0, rec, doc, 0.0)

    assert f"pred:{other}" in label


def test_sheet_label_says_none_below_threshold():
    from agent_edit.backends.pmt import backend as PB

    be = PB.make_backend("cub-10")
    rec = {"gt_by_cid": {cid: [] for cid in be.class_ids}}
    doc = {"instances": [{"score": 0.1, "class_name": be.names[be.class_ids[0]]}]}

    label = be._sheet_label(0, rec, doc, 1.0)

    assert "pred:none" in label


def test_worst_k_sorts_ascending():
    from agent_edit.harness.overlays import worst_k

    assert worst_k([0.9, 0.1, 0.5, 0.0], 2) == [3, 1]
    assert worst_k([1.0, 1.0], 5) == [0, 1]
