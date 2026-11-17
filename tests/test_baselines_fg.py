"""cv-protocol PMT baselines: the fast scorer gives the official counts; folds; CLI dispatch."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mask_utils = pytest.importorskip("pycocotools.mask")

K, NQ, D, HW = 3, 200, 16, (60, 80)


def _rle(x, y, w, h):
    x, y = max(0, x), max(0, y)
    r = mask_utils.frPyObjects(np.array([[x, y, w, h]], np.float64), *HW)[0]
    return {"size": list(HW), "counts": r["counts"].decode()}


def _world(seed, n_img=15):
    """One object per image, some queries on it and the rest anywhere, like a fine-grained set."""
    g = np.random.RandomState(seed)
    recs, entries = [], {}
    for i in range(n_img):
        gx, gy = g.randint(0, 40), g.randint(0, 30)
        rles = [_rle(gx + g.randint(-8, 9), gy + g.randint(-8, 9), 30 + g.randint(-6, 7), 25 + g.randint(-6, 7))
                if q < 40 else _rle(g.randint(0, 70), g.randint(0, 50), g.randint(3, 30), g.randint(3, 30))
                for q in range(NQ)]
        key = f"img{i}"
        entries[key] = {"q": torch.from_numpy(g.randn(NQ, D)).float(), "rles": rles, "hw": HW,
                        "mscore": torch.from_numpy(g.uniform(0.4, 1.0, NQ)).float()}
        recs.append({"image_key": key, "gt_by_cid": {i % K + 1: [_rle(gx, gy, 30, 25)]}, "class_ids": [i % K + 1]})
    return recs, entries


def _world_multi(seed, n_img=15, max_g=3):
    """1..max_g true instances per image at separated cells (xview-ships-10's multi-vessel
    chips), each with its own pool of near-queries, plus background queries, like _world but
    with several objects and not all of the same class."""
    g = np.random.RandomState(seed)
    recs, entries = [], {}
    cells = [(2, 2), (45, 2), (2, 32), (45, 32)]  # corners, far enough apart not to overlap
    for i in range(n_img):
        gi = 1 + (i % max_g)
        slots = [cells[j] for j in g.choice(len(cells), size=gi, replace=False)]
        classes = [int(g.randint(0, K)) + 1 for _ in range(gi)]
        near_per = NQ // (gi + 1)
        rles, insts = [], []
        for (gx, gy), cid in zip(slots, classes):
            for _ in range(near_per):
                rles.append(_rle(gx + g.randint(-3, 4), gy + g.randint(-3, 4), 15 + g.randint(-3, 4), 15 + g.randint(-3, 4)))
            insts.append((cid, _rle(gx, gy, 15, 15)))
        while len(rles) < NQ:
            rles.append(_rle(g.randint(0, 70), g.randint(0, 50), g.randint(3, 15), g.randint(3, 15)))
        g.shuffle(rles)  # near-queries aren't bunched by instance order
        key = f"mimg{i}"
        entries[key] = {"q": torch.from_numpy(g.randn(NQ, D)).float(), "rles": rles, "hw": HW,
                        "mscore": torch.from_numpy(g.uniform(0.4, 1.0, NQ)).float()}
        gt_by_cid: dict = {}
        for cid, r in insts:
            gt_by_cid.setdefault(cid, []).append(r)
        # class_ids lists every class the set has (data.splits.records' own convention), not just
        # the ones this image's GT happens to hit -- a fold can otherwise miss a class entirely.
        recs.append({"image_key": key, "gt_by_cid": gt_by_cid, "class_ids": list(range(1, K + 1))})
    return recs, entries


def _heads(seed):
    g = torch.Generator().manual_seed(seed)
    lin = {"class_rows.weight": 2 * torch.randn(K + 1, D, generator=g), "class_rows.bias": torch.zeros(K + 1)}
    lin["class_rows.bias"][-1] = -2.0
    keys = torch.randn(30, D, generator=g)
    tip = {**lin, "tip_keys": keys / keys.norm(dim=1, keepdim=True), "tip_alpha": torch.tensor(2.0),
           "tip_beta": torch.tensor(3.0), "tip_vals": torch.nn.functional.one_hot(torch.arange(30) % (K + 1)).float()}
    cos = {"class_rows.weight": torch.zeros(K + 1, D), "class_rows.bias": torch.tensor([0.0, 0.5, -0.5, -3.0]),
           "cos_rows": torch.randn(K + 1, D, generator=g), "cos_scale": torch.tensor(20.0)}
    sparse = {k: v.clone() for k, v in lin.items()}
    sparse["class_rows.bias"][-1] = 18.0  # few kept predictions, so some objects are missed
    both = {**tip, "cos_rows": cos["cos_rows"], "cos_scale": cos["cos_scale"]}
    return {"linear": lin, "sparse": sparse, "tip": tip, "cos": cos, "tip+cos": both}


@pytest.fixture(params=[True, False], ids=["box", "mask"])
def toy(request, monkeypatch):
    from agent_edit import data
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import cache as C

    recs, entries = _world(0)
    monkeypatch.setattr(data, "class_table", lambda p: [{"class_id": c + 1, "class_name": f"c{c}"} for c in range(K)])
    monkeypatch.setattr(C, "source_root", lambda p, s="pmt": Path("unused"))
    monkeypatch.setattr(C, "load", lambda root, key: entries[key])
    monkeypatch.setattr(PB, "BOX_PROJECTS", {"toyfg"} if request.param else set())
    return "toyfg", recs, entries


def _official(project, recs, entries, w):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.metrics.f1 import counts

    cids = [c + 1 for c in range(K)]
    names = {c: f"c{c - 1}" for c in cids}
    docs = {}
    for r in recs:
        e = entries[r["image_key"]]
        p = PB.predictions(e, w, PB.SCORE_THRESHOLD, project in PB.BOX_PROJECTS)
        docs[r["image_key"]] = PB.to_doc(project, r, e["hw"], p, names, cids, "train")
    return counts(recs, docs, PB.SCORE_THRESHOLD)[0].sum(0)


def test_fast_scorer_gives_the_official_counts(toy):
    from agent_edit.baselines.common import Images, fast_counts, fast_f1, folds
    from agent_edit.metrics.f1 import f1_from_counts

    project, recs, entries = toy
    ims = Images(project, recs)
    n_pred = 0
    for name, w in _heads(1).items():
        got = fast_counts(ims, w)
        assert (got == _official(project, recs, entries, w)).all(), name
        assert fast_f1(ims, w) == f1_from_counts(_official(project, recs, entries, w))[0], name
        n_pred += int(got[:, :2].sum())
        for ho in folds(len(recs), K):  # the held-out folds the pick is scored on
            sub = [recs[i] for i in ho]
            assert (fast_counts(ims.sub(ho.tolist()), w) == _official(project, sub, entries, w)).all(), name
    assert n_pred > 0  # the heads must predict something, or the check is empty


def test_images_supports_multiple_instances_per_image(monkeypatch):
    """xview-ships-10's chips can hold several vessels: Images accepts that, and fast_counts still
    matches the official scorer."""
    from agent_edit import data
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import cache as C
    from agent_edit.baselines.common import Images, fast_counts, fast_f1, folds
    from agent_edit.metrics.f1 import f1_from_counts

    recs, entries = _world_multi(0)
    monkeypatch.setattr(data, "class_table", lambda p: [{"class_id": c + 1, "class_name": f"c{c}"} for c in range(K)])
    monkeypatch.setattr(C, "source_root", lambda p, s="pmt": Path("unused"))
    monkeypatch.setattr(C, "load", lambda root, key: entries[key])
    monkeypatch.setattr(PB, "BOX_PROJECTS", set())
    project = "toyfg_multi"

    ims = Images(project, recs)
    assert not ims.single
    assert len(ims.inst_gt) > len(recs)  # at least one image really does hold more than one instance

    n_pred = 0
    for name, w in _heads(2).items():
        got = fast_counts(ims, w)
        assert (got == _official(project, recs, entries, w)).all(), name
        assert fast_f1(ims, w) == f1_from_counts(_official(project, recs, entries, w))[0], name
        n_pred += int(got[:, :2].sum())
        for ho in folds(len(recs), K):
            sub = [recs[i] for i in ho]
            assert (fast_counts(ims.sub(ho.tolist()), w) == _official(project, sub, entries, w)).all(), name
    assert n_pred > 0


@pytest.mark.parametrize("n", [5, 10, 20])
def test_folds_hold_out_the_same_share_of_every_class(n):
    from agent_edit.baselines.common import FOLDS, folds

    k = 10
    fl = folds(n * k, k)
    assert len(fl) == FOLDS
    assert sorted(np.concatenate(fl).tolist()) == list(range(n * k))
    for f in fl:
        assert np.bincount(f % k, minlength=k).tolist() == [n // FOLDS] * k


def test_cli_dispatch_c2_c5f(monkeypatch):
    from agent_edit.baselines import decoder_ft
    from agent_edit.cli import main

    calls = []
    monkeypatch.setattr(decoder_ft, "run_fold", lambda *a, **k: calls.append(("fold", a, k)))
    monkeypatch.setattr(decoder_ft, "run", lambda *a, **k: calls.append(("c5f", a, k)))
    assert main(["baselines", "c5f", "cub-10", "--n", "5", "--fold", "3"]) == 0
    assert main(["baselines", "c5f", "cub-10", "--n", "5"]) == 0
    assert main(["baselines", "c5f", "cub-10", "--n", "5", "--order", "2"]) == 0
    assert calls == [("fold", ("cub-10", 0, 5, 3, None), {"order": 0}),
                     ("c5f", ("cub-10", 0, 5, None), {"order": 0}),
                     ("c5f", ("cub-10", 0, 5, None), {"order": 2})]
    for args, msg in ((["c2", "cub-10", "--order", "1"], "--order is for c5f"),
                      (["c5f", "cub-10", "--kd-weight", "3"], "--kd-weight is for lwf"),
                      (["lwf", "cub-10"], "needs --kd-weight")):
        with pytest.raises(SystemExit, match=msg):
            main(["baselines", *args])


def test_cli_dispatch_lwf(monkeypatch):
    from agent_edit.baselines import decoder_ft
    from agent_edit.cli import main

    calls = []
    monkeypatch.setattr(decoder_ft, "run_lwf_final", lambda *a, **k: calls.append((a, k)))
    assert main(["baselines", "lwf", "cub-10", "--n", "5", "--kd-weight", "3"]) == 0
    assert calls == [(("cub-10", 0, 5, 3.0, None), {"order": 0})]


def test_pair_costs_matmul_form_matches_elementwise(monkeypatch):
    from agent_edit.baselines import decoder_ft

    g = torch.Generator().manual_seed(0)
    ml = torch.randn(7, 12, 12, generator=g) * 4
    gm = (torch.rand(5, 12, 12, generator=g) > 0.6).float()
    dense = decoder_ft.pair_costs(ml, gm)
    monkeypatch.setattr(decoder_ft, "DENSE_MATCH_MAX", 0)
    for a, b in zip(dense, decoder_ft.pair_costs(ml, gm)):
        assert a.shape == (7, 5) and torch.allclose(a, b, atol=1e-5)
