import numpy as np
import pytest

from agent_edit.metrics import docs as md
from agent_edit.metrics.bootstrap import bootstrap_diff, f1_ci
from agent_edit.metrics.f1 import class_scores, counts, f1_from_counts, image_stats, per_image_f1
from agent_edit.metrics.match import greedy_match, prf, rle_of

H, W = 20, 20


def box(y0, x0, y1, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def rle(m):
    return md.encode_masks([m])[0]


def test_rle_of_accepts_str_bytes_and_dense():
    r = rle(box(0, 0, 5, 5))
    assert isinstance(rle_of(r)["counts"], bytes)
    assert rle_of(r) == rle_of({**r, "counts": r["counts"].encode()})
    assert rle_of(box(0, 0, 5, 5))["counts"] == rle_of(r)["counts"]


def test_greedy_match_claims_in_score_order():
    gt = [box(0, 0, 10, 10)]
    preds = [box(0, 0, 10, 9), box(0, 0, 10, 10)]  # the lower score is the perfect mask
    tp, fp, fn, dets, ious = greedy_match(preds, [0.4, 0.9], gt, 0.5)
    assert (tp, fp, fn) == (1, 1, 0)
    assert dets == [(0.9, True), (0.4, False)]
    assert ious == [1.0]


def test_greedy_match_iou_cut_and_empty_cases():
    gt = [box(0, 0, 10, 10)]
    tp, fp, fn, _, _ = greedy_match([box(0, 0, 10, 4)], [1.0], gt, 0.5)  # IoU 0.4
    assert (tp, fp, fn) == (0, 1, 1)
    assert greedy_match([], [], gt, 0.5)[:3] == (0, 0, 1)
    assert greedy_match([box(0, 0, 2, 2)], [0.3], [], 0.5)[:4] == (0, 1, 0, [(0.3, False)])


def test_greedy_match_ties_keep_input_order():
    gt = [box(0, 0, 10, 10)]
    _, _, _, dets, _ = greedy_match([box(0, 0, 10, 10), box(0, 0, 10, 10)], [0.5, 0.5], gt, 0.5)
    assert dets == [(0.5, True), (0.5, False)]


def test_greedy_match_refuses_mixed_canvases():
    with pytest.raises(ValueError):
        greedy_match([np.ones((4, 4), bool)], [1.0], [np.ones((5, 5), bool)], 0.5)


def test_prf():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(2, 2, 0)
    assert (p, r) == (0.5, 1.0) and f == pytest.approx(2 / 3)


def _records_and_docs():
    """Two images, classes 1 and 2. Image a: class-1 hit + class-2 false alarm; image b: class-1 miss."""
    names = {1: "car", 2: "bus"}
    a_gt, b_gt = box(0, 0, 10, 10), box(10, 10, 20, 20)
    recs = [
        {"image_key": "a", "class_ids": [1, 2], "class_names": names, "gt_by_cid": {1: [rle(a_gt)]}, "hw": [H, W]},
        {"image_key": "b", "class_ids": [1, 2], "class_names": names, "gt_by_cid": {1: [rle(b_gt)]}, "hw": [H, W]},
    ]
    inst_a = md.make_instances(1, "car", [0.9], [[0, 0, 10, 10]], [a_gt]) + \
        md.make_instances(2, "bus", [0.8], [[0, 0, 5, 5]], [box(0, 0, 5, 5)]) + \
        md.make_instances(1, "car", [0.2], [[0, 0, 3, 3]], [box(15, 0, 18, 3)])
    docs = {"a": md.make_doc("p", "a", (H, W), inst_a), "b": md.make_doc("p", "b", (H, W), [])}
    return recs, docs


def test_class_scores_and_counts_agree():
    recs, docs = _records_and_docs()
    f1, per_class = class_scores(recs, docs, min_score=0.5)
    C, cids = counts(recs, docs, 0.5)
    assert cids == [1, 2]
    assert C.tolist() == [[[1, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 0, 0]]]
    car_f1 = 2 * 1 / (2 * 1 + 0 + 1)
    assert per_class[1]["f1"] == pytest.approx(car_f1)
    assert per_class[2]["f1"] == 0.0
    assert f1 == pytest.approx(car_f1 / 2)
    assert f1_from_counts(C)[0] == pytest.approx(f1)


def test_min_score_drops_low_predictions():
    recs, docs = _records_and_docs()
    C_all, _ = counts(recs, docs, 0.0)
    assert C_all[0, 0].tolist() == [1, 1, 1]  # the 0.2 car mask is an extra false positive


def test_per_image_f1():
    recs, docs = _records_and_docs()
    C, _ = counts(recs, docs, 0.5)
    np.testing.assert_allclose(per_image_f1(C), [0.5, 0.0])
    assert per_image_f1(np.zeros((1, 2, 3), np.int64)).tolist() == [1.0]
    assert image_stats(recs[0], docs["a"], 0.5)["f1"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_zero_for_equal_heads():
    rng = np.random.RandomState(0)
    C = rng.randint(0, 5, (30, 3, 3))
    C[..., 2] = C[..., 0] + rng.randint(0, 3, (30, 3))
    b = bootstrap_diff(C, C, n=200)
    assert b["diff"] == 0.0 and b["ci95"] == [0.0, 0.0]
    better = C.copy()
    better[..., 1] = 0
    b1, b2 = bootstrap_diff(C, better, n=200), bootstrap_diff(C, better, n=200)
    assert b1 == b2 and b1["diff"] > 0 and b1["ci95"][0] > 0
    lo, hi = f1_ci(C, n=200)
    assert lo <= f1_from_counts(C)[0] <= hi


def test_train_report_scores_exactly_the_given_images():
    from agent_edit.metrics.f1 import train_report

    names = {1: "x", 2: "y"}
    recs = [{"image_key": k, "class_ids": [1, 2], "class_names": names, "gt_by_cid": {c: [rle(box(0, 0, 10, 10))], 3 - c: []},
             "hw": [H, W]} for k, c in (("a", 1), ("b", 2))]
    docs = {r["image_key"]: md.make_doc("p", r["image_key"], (H, W), md.make_instances(
        c, names[c], [0.9], [[0, 0, 10, 10]], [box(0, 0, 10, 10)])) for r, c in zip(recs, (1, 2))}
    rep = train_report("p", recs, docs, 2, 0, 0.5)
    assert rep["macro_f1"] == 1.0 and rep["n_gt"] == 2
    assert {k: v["n_gt"] for k, v in rep["per_class"].items()} == {"x": 1, "y": 1}
