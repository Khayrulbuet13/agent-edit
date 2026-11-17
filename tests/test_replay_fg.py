"""Cross-set replay of the fine-grained PMT sets: the class-row swap in analysis/replay.py."""

import torch

from agent_edit.analysis.replay import FG_PARENT_COCO_ROW, _fg_parent_fix


def _weights(k=11, d=4):
    return {"class_rows.weight": torch.arange(k * d, dtype=torch.float32).reshape(k, d),
            "class_rows.bias": torch.arange(k, dtype=torch.float32)}


def test_same_project_is_a_no_op():
    """Same-project replay sees the untouched dict, so an existing recipe keeps reproducing its own
    saved edit."""
    w = _weights()
    out = _fg_parent_fix("agent_edit.backends.pmt.backend:make_backend", "stanford-cars-10", "stanford-cars-10", w)
    assert out is w


def test_unknown_project_is_a_no_op():
    w = _weights()
    out = _fg_parent_fix("agent_edit.backends.pmt.backend:make_backend", "stanford-cars-10", "no-such-set", w)
    assert out is w
    out = _fg_parent_fix("agent_edit.backends.pmt.backend:make_backend", "no-such-set", "stanford-cars-10", w)
    assert out is w


def test_non_pmt_backend_is_a_no_op():
    w = _weights()
    out = _fg_parent_fix("agent_edit.backends.fake:FakeBackend", "stanford-cars-10", "cub-10", w)
    assert out is w


def test_cross_set_swaps_only_the_source_projects_own_slot(monkeypatch):
    """Cars reads row 2 (COCO car); replayed on cub-10, row 2 must become cub-10's own COCO row
    (bird, continuous index 14), not stay COCO 'car' and not touch any other row."""
    coco_w = torch.arange(80 * 4, dtype=torch.float32).reshape(80, 4)
    coco_b = torch.arange(80, dtype=torch.float32)

    def fake_load(device="cpu"):
        return {"class_head.weight": coco_w, "class_head.bias": coco_b}

    monkeypatch.setattr("agent_edit.backends.pmt.head.load_head_weights", fake_load)

    w = _weights()
    out = _fg_parent_fix("agent_edit.backends.pmt.backend:make_backend", "stanford-cars-10", "cub-10", w)
    assert out is not w  # a fresh dict, original untouched
    assert torch.equal(out["class_rows.weight"][2], coco_w[14])
    assert out["class_rows.bias"][2] == coco_b[14]
    for row in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10):
        assert torch.equal(out["class_rows.weight"][row], w["class_rows.weight"][row])
        assert out["class_rows.bias"][row] == w["class_rows.bias"][row]
    # the input dict itself is never mutated
    assert torch.equal(w["class_rows.weight"][2], torch.arange(8.0, 12.0))


def test_aircraft_row_is_out_of_bounds_for_itself_as_a_target_of_bird_or_cat():
    """cub-10/pets-10's own COCO parent row (14/15) sits outside the 11-row class head, so a
    recipe anchored on it (none exist) couldn't be moved to another set either -- make sure we just skip it
    instead of raising."""
    w = _weights()
    out = _fg_parent_fix("agent_edit.backends.pmt.backend:make_backend", "cub-10", "stanford-cars-10", w)
    assert out is w


def test_known_parent_rows_unchanged():
    """Pin the 4 project -> COCO-row values (plus their -mix5 twins, same row) and xview-ships-10;
    if COCO's class order ever changes upstream, this test (not a silent wrong replay) should be
    the thing that breaks."""
    assert FG_PARENT_COCO_ROW == {"stanford-cars-10": 2, "cub-10": 14, "pets-10": 15, "aircraft-10": 4,
                                  "stanford-cars-10-mix5": 2, "cub-10-mix5": 14, "pets-10-mix5": 15,
                                  "aircraft-10-mix5": 4, "xview-ships-10": 8}
