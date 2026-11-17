"""Full decoder fine-tuning (decoder_ft) grid shape and pick rule. CPU, fast -- no GPU, no cache needed."""

import pytest
import torch

from agent_edit.baselines import decoder_ft as D


def test_grid_spans_the_documented_range():
    assert min(D.LRS) == 1e-5
    assert max(D.LRS) == 1e-3
    assert min(D.EPOCHS) == 1
    assert max(D.EPOCHS) == 24
    assert len(D.GRID) == len(D.LRS) * len(D.EPOCHS)
    assert len(set(D.GRID)) == len(D.GRID)  # every (lr, epoch) pair is distinct


def test_pick_takes_the_best_cv_score():
    cv = {str(g): 0.0 for g in D.GRID}
    winner = D.GRID[5]
    cv[str(winner)] = 0.9
    assert D.pick(cv) == winner


def test_pick_breaks_ties_to_the_earlier_grid_point():
    cv = {str(g): 0.0 for g in D.GRID}
    cv[str(D.GRID[3])] = 0.5
    cv[str(D.GRID[9])] = 0.5  # same score, later in GRID
    assert D.pick(cv) == D.GRID[3]


def test_tag_order_0_keeps_the_pre_sweep_name():
    assert D.tag(seed=0, n=5) == "C5F_s0_N5"
    assert D.tag(seed=1, n=20, order=0) == "C5F_s1_N20"


def test_tag_nonzero_order_is_distinct_per_order():
    names = {D.tag(seed=0, n=5, order=o) for o in (0, 1, 2)}
    assert len(names) == 3
    assert "C5F_s0_N5" in names
    assert "C5F_s0_o1_N5" in names and "C5F_s0_o2_N5" in names


def test_lwf_tag_names_the_lambda_and_order():
    assert D.lwf_tag(seed=0, n=5, kd_weight=3.0) == "LWF_s0_l3_N5"
    assert D.lwf_tag(seed=1, n=20, kd_weight=0.5, order=2) == "LWF_s1_l0p5_o2_N20"
    names = {D.lwf_tag(0, 5, lam) for lam in D.LWF_LAMBDAS}
    assert len(names) == len(D.LWF_LAMBDAS)  # every lambda gets its own file


def test_kd_loss_is_zero_when_matching_the_stock_output():
    masks = torch.randn(2, 4, 4)
    cls = torch.randn(2, 81)
    assert D.kd_loss(masks, cls, masks, cls).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_grows_with_mismatch():
    g = torch.Generator().manual_seed(0)
    masks = torch.randn(2, 4, 4, generator=g)
    cls = torch.randn(2, 81, generator=g)
    near = D.kd_loss(masks, cls, masks + 0.01, cls + 0.01).item()
    far = D.kd_loss(masks, cls, masks + 5.0, cls + 5.0).item()
    assert 0.0 < near < far
