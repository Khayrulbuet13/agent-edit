"""The start="parent" backend argument (start_weights, make_backend, describe).

Fine-grained "-10" sets normally start every class row as the first 10 COCO rows. start="parent"
puts the project's real COCO parent row (FG_PARENT_COCO_ROW) in slot 0 instead -- the same
condition cars already gets by coincidence (COCO "car" = row 2, inside the first-10 slice). Where
that coincidence already holds (cars: car=2, aircraft: airplane=4) start="parent" is a no-op,
identical to start="coco". Where it doesn't (cub: bird=14, pets: cat=15) slot 0 becomes the parent
row; slots 1..K-1 and no-object stay start="coco"'s. The default start="coco" gives
the stock rows.
"""

import pytest

from conftest import need_manifest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _manifests():
    need_manifest("stanford-cars-10", "cub-10", "pets-10", "aircraft-10")


def _coco_head(rows=81, d=4):
    return {"class_head.weight": torch.arange(rows * d, dtype=torch.float32).reshape(rows, d),
            "class_head.bias": torch.arange(rows, dtype=torch.float32)}


def test_start_weights_default_is_unchanged(monkeypatch):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    coco = _coco_head()
    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": coco)
    w = PB.start_weights("cub-10")  # start not given -> "coco"
    rows = PB.slot_rows("cub-10") + [H.NO_OBJECT]
    assert torch.equal(w["class_rows.weight"], coco["class_head.weight"][rows])
    assert torch.equal(w["class_rows.bias"], coco["class_head.bias"][rows])


@pytest.mark.parametrize("project", ["stanford-cars-10", "aircraft-10"])
def test_start_parent_is_a_no_op_where_the_parent_row_is_already_in_slot(project, monkeypatch):
    """cars (car=row2) and aircraft (airplane=row4) already have their parent inside the first-10
    slice start='coco' uses, so start='parent' must give the identical tensors."""
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    coco_w = PB.start_weights(project, start="coco")
    parent_w = PB.start_weights(project, start="parent")
    assert torch.equal(coco_w["class_rows.weight"], parent_w["class_rows.weight"])
    assert torch.equal(coco_w["class_rows.bias"], parent_w["class_rows.bias"])


@pytest.mark.parametrize("project,parent_name", [("cub-10", "bird"), ("pets-10", "cat")])
def test_start_parent_replaces_only_slot0(project, parent_name, monkeypatch):
    """cub (bird=row14) and pets (cat=row15) never land inside the first-10 slice, so slot 0
    becomes the parent row and slots 1..K-1 (+ no-object) stay start='coco''s."""
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    coco = _coco_head()
    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": coco)
    coco_w = PB.start_weights(project, start="coco")
    parent_w = PB.start_weights(project, start="parent")
    parent_row = coco["class_head.weight"][PB.FG_PARENT_COCO_ROW[project]]
    assert torch.equal(parent_w["class_rows.weight"][0], parent_row)
    assert parent_w["class_rows.bias"][0] == coco["class_head.bias"][PB.FG_PARENT_COCO_ROW[project]]
    K = len(PB.slot_rows(project))
    for j in range(1, K + 1):  # slots 1..K-1, then no-object (index K)
        assert torch.equal(parent_w["class_rows.weight"][j], coco_w["class_rows.weight"][j])
        assert parent_w["class_rows.bias"][j] == coco_w["class_rows.bias"][j]


def test_start_parent_without_a_parent_row_raises(monkeypatch):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    monkeypatch.delitem(PB.FG_PARENT_COCO_ROW, "pets-10")
    with pytest.raises(ValueError, match="start='parent'"):
        PB.start_weights("pets-10", start="parent")


def test_unknown_start_raises(monkeypatch):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    with pytest.raises(ValueError, match="unknown start"):
        PB.start_weights("cub-10", start="bogus")


def test_make_backend_threads_start_to_initial_weights(monkeypatch):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    coco = _coco_head()
    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": coco)
    be = PB.make_backend("cub-10", start="parent")
    assert be.start == "parent"
    w = be.initial_weights()
    bird = coco["class_head.weight"][PB.FG_PARENT_COCO_ROW["cub-10"]]
    assert torch.equal(w["class_rows.weight"][0], bird)
    # default (no start arg) still gives the first-K COCO rows
    be2 = PB.make_backend("cub-10")
    assert be2.start == "coco"
    assert torch.equal(be2.initial_weights()["class_rows.weight"], PB.start_weights("cub-10")["class_rows.weight"])


def test_describe_default_text_is_unchanged(monkeypatch):
    """the default start='coco' text carries the stock-rows sentence."""
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    monkeypatch.setattr(PB.PmtHeadBackend, "n_images", lambda self: 20)  # the image count is not under test
    be = PB.make_backend("cub-10")
    text = be.describe()
    assert ("The class rows you start from are rows of its COCO class head (for COCO-like projects, "
            "the matching COCO classes; otherwise the first 10 COCO classes, whose meaning is "
            "unrelated to this project) and the no-object row of that head.") in text


def test_describe_parent_text_names_slot0(monkeypatch):
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    monkeypatch.setattr(PB.PmtHeadBackend, "n_images", lambda self: 20)  # the image count is not under test
    be = PB.make_backend("cub-10", start="parent")
    text = be.describe()
    assert "Row 0 starts as a copy of COCO's \"bird\" row" in text
    assert "calls every one of them row 0's class" in text
    assert "whose meaning is unrelated to this project" in text
    # and it must NOT read as the (different) start='coco' sentence
    assert "otherwise the first 10 COCO classes" not in text


def test_describe_parent_text_is_the_coco_text_where_parent_is_a_no_op(monkeypatch):
    """cars: start='parent' changes no tensor, so the description must say the same thing as
    start='coco' (no "row 0" sentence that doesn't match what actually happened)."""
    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import head as H

    monkeypatch.setattr(H, "load_head_weights", lambda device="cpu": _coco_head())
    monkeypatch.setattr(PB.PmtHeadBackend, "n_images", lambda self: 20)  # the image count is not under test
    coco_text = PB.make_backend("stanford-cars-10").describe()
    parent_text = PB.make_backend("stanford-cars-10", start="parent").describe()
    assert coco_text == parent_text


def test_score_run_reads_start_from_harness_json(tmp_path, monkeypatch):
    """score_run must use the run's own start/source (harness.json backend_args), not the
    module default, so a parent-started run is scored against the head it actually started from."""
    import json

    from agent_edit.backends.pmt import backend as PB
    from agent_edit.backends.pmt import evaluate as EV

    run = tmp_path / "cub-10" / "pmt_s0_N5_B25_sol"
    run.mkdir(parents=True)
    (run / "harness.json").write_text(json.dumps({"backend_args": {"project": "cub-10", "seed": "0",
                                                                     "n_shot": "5", "start": "parent"}}))

    seen = {}

    def fake_start_weights(project, source=None, start="coco"):
        seen["start_weights"] = (project, source, start)
        return {"class_rows.weight": torch.zeros(11, 4), "class_rows.bias": torch.zeros(11)}

    def fake_make_backend(project, seed=None, n_shot=None, source=None, start="coco", **_):
        seen["make_backend"] = (project, source, start)

        class Stub:
            def _ensure(self):
                self._recs = [0, 0]

            def generate(self, weights, n):
                return {"macro_f1": 0.0}

        return Stub()

    def fake_evaluate(project, weights, source=None, start="coco", **_):
        seen["evaluate"] = (project, source, start)
        return {"macro_f1": 0.0, "ref_macro_f1": 0.0, "bootstrap_minus_ref": {"ci95": [0, 0]},
                "recall_over_ceiling": None}

    monkeypatch.setattr(PB, "start_weights", fake_start_weights)
    monkeypatch.setattr(PB, "make_backend", fake_make_backend)
    monkeypatch.setattr(EV, "evaluate", fake_evaluate)

    EV.score_run(run)

    assert seen["start_weights"] == ("cub-10", None, "parent")
    assert seen["make_backend"] == ("cub-10", None, "parent")
    assert seen["evaluate"] == ("cub-10", None, "parent")
