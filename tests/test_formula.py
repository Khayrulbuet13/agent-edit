"""The Agent formula and its COCO guard, on synthetic tensors (no project data, no caches)."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from agent_edit.baselines import formula as F  # noqa: E402

K, D, NQ, N_IMG = 3, 8, 12, 9


@pytest.fixture
def stock_rows(monkeypatch):
    """A fake stock head (K class rows + the no-object row), so `start` needs no checkpoint."""
    g = torch.Generator().manual_seed(0)
    W, b = torch.randn(K + 1, D, generator=g), torch.randn(K + 1, generator=g)
    monkeypatch.setattr(F, "start", lambda project: {"class_rows.weight": W.clone(), "class_rows.bias": b.clone()})
    return W, b


def fake_images(project="cub-10", n=N_IMG, seed=1):
    """One true instance per image; query 0 sits on it (IoU 0.9), the rest miss it."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, NQ, D, generator=g)
    iou = torch.zeros(n, NQ)
    iou[:, 0] = 0.9
    gt = torch.arange(n) % K
    q[:, 0] += 3 * torch.nn.functional.one_hot(gt, D).float()  # primaries separate by class and from the rest
    return SimpleNamespace(q=q, mscore=torch.rand(n, NQ, generator=g), inst_img=torch.arange(n), inst_gt=gt,
                           inst_iou=iou, K=K, project=project)


# ---------- grid ----------

def test_grid_sizes_and_gate_kinds():
    per_kind = len(F.T_GATE_GRID) * len(F.S_CLS_GRID) * len(F.K_PCS_GRID)
    assert len(F.grid_for("cub-10")) == per_kind and F.gate_options("cub-10") == ("learned",)
    assert len(F.grid_for("stanford-cars-10")) == 2 * per_kind
    assert {c[0] for c in F.grid_for("stanford-cars-10")} == {"stock", "learned"}
    assert F.grid_for("cub-10")[0] == ("learned", 0.0, 1.0, 0)


def test_mix5_sets_follow_their_base_set():
    assert F.gate_options("aircraft-10-mix5") == ("stock", "learned")
    assert F.gate_options("pets-10-mix5") == ("learned",)
    assert F.PARENT_ROW["xview-ships-10"] == 8


# ---------- _shared / _combine ----------

@pytest.mark.parametrize("mode", ["learned", "stock"])
def test_shared_shapes(stock_rows, monkeypatch, mode):
    monkeypatch.setitem(F.PARENT_ROW, "cub-10", 1)
    ims = fake_images()
    ims.mscore[:] = 1.0  # lets the stock-confident (score >= 0.5) misses exist
    sh = F._shared(ims, mode, 5)
    assert sh is not None
    assert sh["present"] == [0, 1, 2]
    assert sh["Dc_base"].shape == (3, D) and sh["v"].shape == (D,) and sh["m0"].shape == (D,)
    assert sh["Dc_base"].mean(0).abs().max() < 1e-12  # centred over classes
    assert sh["max_k_cap"] == min(5, (N_IMG - 3) // 3)
    assert sh["Vh_full"].shape == (sh["max_k_cap"], D)


def test_gate_scale_by_kind(stock_rows):
    ims = fake_images()
    sh = F._shared(ims, "learned", 0)
    Xp = ims.q[:, 0].double()
    assert float((Xp @ sh["v"]).std()) == pytest.approx(1.0)
    assert sh["Vh_full"] is None


def test_combine_zero_class_gain_rows_differ_only_by_gate(stock_rows):
    W, b = stock_rows
    ims = fake_images()
    sh = F._shared(ims, "learned", 5)
    w = F._combine(ims, sh, t_gate=4.0, s_cls=0.0, k_pcs=0)
    rows, bias = w["class_rows.weight"][:K], w["class_rows.bias"][:K]
    assert torch.allclose(rows, rows[0].expand(K, D)) and torch.allclose(bias, bias[0].expand(K))
    assert torch.allclose(rows[0].double(), W[K].double() + 4.0 * sh["v"], atol=1e-5)
    assert torch.equal(w["class_rows.weight"][K], W[K])  # no-object row untouched


def test_combine_zero_removed_directions_keeps_class_means(stock_rows):
    W, _ = stock_rows
    ims = fake_images()
    sh = F._shared(ims, "learned", 5)
    w = F._combine(ims, sh, t_gate=0.0, s_cls=2.0, k_pcs=0)
    want = W[K].double() + 2.0 * sh["Dc_base"]
    assert torch.allclose(w["class_rows.weight"][:K].double(), want, atol=1e-5)


def test_combine_removed_directions_are_orthogonal(stock_rows):
    ims = fake_images()
    sh = F._shared(ims, "learned", 5)
    k = sh["max_k_cap"]
    assert k > 0
    w = F._combine(ims, sh, t_gate=0.0, s_cls=1.0, k_pcs=50)  # asks for more than the cap allows
    Dc = (w["class_rows.weight"][:K].double() - stock_rows[0][K].double())
    assert (Dc @ sh["Vh_full"][:k].T).abs().max() < 1e-5


def test_combine_leaves_absent_classes_at_stock(stock_rows):
    W, b = stock_rows
    ims = fake_images()
    ims.inst_gt = ims.inst_gt % 2  # class 2 has no image
    sh = F._shared(ims, "learned", 0)
    assert sh["present"] == [0, 1]
    w = F._combine(ims, sh, 2.0, 1.0, 0)
    assert torch.equal(w["class_rows.weight"][2], W[2]) and w["class_rows.bias"][2] == b[2]


def test_fit_formula_skips_when_no_gate_can_be_made(stock_rows):
    ims = fake_images()
    ims.inst_iou[:] = 0.0  # nothing hits an object
    w, info = F.fit_formula(ims, ("learned", 1.0, 1.0, 0))
    assert "skipped" in info and torch.equal(w["class_rows.weight"], stock_rows[0])


def test_shared_uses_the_instance_rows_of_multi_instance_images(stock_rows):
    ims = fake_images(n=4)
    ims.inst_img = torch.tensor([0, 0, 1, 2, 3])  # image 0 holds two instances
    ims.inst_gt = torch.tensor([0, 1, 2, 0, 1])
    ims.inst_iou = torch.zeros(5, NQ)
    ims.inst_iou[:, 0] = 0.9
    ims.inst_iou[1, 0], ims.inst_iou[1, 5] = 0.0, 0.9  # the second instance sits on another query
    sh = F._shared(ims, "learned", 0)
    assert sh["present"] == [0, 1, 2]


# ---------- guard ----------

def walk(cv, drops, bound=1.0):
    grid = [("learned", float(i), 1.0, 0) for i in range(len(cv))]
    cvd = {str(g): s for g, s in zip(grid, cv)}
    by_cfg = {g: d for g, d in zip(grid, drops)}
    return F.guard_walk(grid, cvd, lambda cfg: (cfg, {"n": 1}), lambda w: by_cfg[w], bound), grid


def test_guard_takes_first_candidate_in_cv_order_that_passes():
    (row, passed, checked), grid = walk([0.2, 0.9, 0.5, 0.7], [0.1, 5.0, 3.0, 0.4])
    assert passed and checked == 2  # 0.9 fails, 0.7 passes
    assert row["picked"] == list(grid[3]) and row["cv_f1"] == 0.7 and row["retain20_drop"] == 0.4


def test_guard_ties_keep_grid_order():
    (row, passed, checked), grid = walk([0.5, 0.5, 0.5], [9.0, 0.0, 0.0])
    assert passed and checked == 2 and row["picked"] == list(grid[1])


def test_guard_falls_back_to_smallest_drop():
    (row, passed, checked), grid = walk([0.9, 0.8, 0.7], [4.0, 2.5, 3.0])
    assert not passed and checked == 3 and row["picked"] == list(grid[1])


def test_guard_bound_is_inclusive():
    (_, passed, checked), _ = walk([0.5], [1.0])
    assert passed and checked == 1


# ---------- cell and command line ----------

class FakeGuard:
    def drop(self, project, w):
        return 0.25


@pytest.fixture
def cell_env(monkeypatch):
    """run_cell with data, fits and scoring replaced; records what it was asked to load."""
    from agent_edit.backends.pmt import evaluate as E

    seen = {"splits": [], "cv_for": []}
    monkeypatch.setattr(F.data, "load_split", lambda project, split, n=None, seed=0: seen["splits"].append((n, seed)) or [n])
    monkeypatch.setattr(F, "Images", lambda project, recs: SimpleNamespace(n=recs[0]))

    def cv(ims):
        seen["cv_for"].append(ims.n)
        grid = F.grid_for("cub-10")
        return grid[7], {str(g): (1.0 if i == 7 else 0.1) for i, g in enumerate(grid)}

    monkeypatch.setattr(F, "cv_pick_formula", cv)
    monkeypatch.setattr(F, "fit_formula_final", lambda ims, cfg: ({"class_rows.weight": torch.zeros(1)}, {"cfg_n": ims.n}))
    monkeypatch.setattr(E, "evaluate", lambda project, w, **kw: {"macro_f1": 0.5, "macro_f1_ci95": [0.4, 0.6]})
    return seen


def test_cell_writes_the_stored_fields(cell_env, tmp_path):
    rec = F.run_cell("cub-10", 5, 2, tmp_path, guard=FakeGuard())
    saved = json.loads((tmp_path / "cub-10_s2_N5.json").read_text())
    for k in ("project", "seed", "n", "picked", "cv_f1", "retain20_drop", "guard_passed", "n_cv_points_checked",
              "eval_f1", "eval_ci95", "fit_info", "seconds"):
        assert k in saved
    assert (saved["picked"], saved["cv_f1"], saved["retain20_drop"]) == (list(F.grid_for("cub-10")[7]), 1.0, 0.25)
    assert saved["guard_passed"] is True and saved["n_cv_points_checked"] == 1
    assert saved["eval_f1"] == 0.5 and saved["carried_from_n2"] is None
    assert (tmp_path / "cub-10_s2_N5.pt").exists() and rec["n"] == 5
    assert cell_env["cv_for"] == [5]


def test_n1_ranks_the_grid_with_the_n2_cv_scores(cell_env, tmp_path):
    rec = F.run_cell("cub-10", 1, 0, tmp_path, guard=FakeGuard())
    assert cell_env["splits"] == [(1, 0), (2, 0)]
    assert cell_env["cv_for"] == [2]  # no CV on the 1-per-class images
    assert rec["fit_info"] == {"cfg_n": 1}  # fitted on the N=1 images
    assert rec["picked"] == list(F.grid_for("cub-10")[7]) and rec["carried_from_n2"] == "cub-10_s0_N2"


def test_parser_and_main(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(F, "run_cell", lambda *a, **k: calls.append(a))
    assert F.main(["pets-10", "--n", "2", "--seed", "3", "--out", str(tmp_path)]) == 0
    assert calls == [("pets-10", 2, 3, tmp_path)]
    a = F.build_parser().parse_args(["cub-10", "--out", "x"])
    assert (a.n, a.seed) == (F.data.N_TRAIN, 0)


def test_out_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        F.main(["cub-10", "--n", "5"])
    assert e.value.code == 2
    assert "--out" in capsys.readouterr().err
