"""PMT backend wiring: the optional lab module, the shared image loader, the guard-cache builder.

No AGENT_EDIT_* variable is needed; the tests clear them.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("pycocotools")

import agent_edit  # noqa: E402

PKG = Path(agent_edit.__file__).parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith("AGENT_EDIT_")]:
        monkeypatch.delenv(k)


def _run_in_copy(tmp_path, drop, code) -> subprocess.CompletedProcess:
    """Run `code` against a copy of the package with the paths in `drop` removed."""
    dst = tmp_path / "src" / "agent_edit"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for d in drop:
        p = dst / d
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_EDIT_")}
    env["PYTHONPATH"] = str(tmp_path / "src")
    return subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_backend_works_without_lab_sets(tmp_path):
    code = """
import sys
import agent_edit
assert agent_edit.__file__.startswith(sys.argv[1]), agent_edit.__file__
from agent_edit.backends.pmt import backend as PB, cache as C
assert PB.LAB is None and PB.SOURCES == {} and PB.COCO_ROWS == {}
assert set(PB.TASKS) == set(PB.FG_PARENT_COCO_ROW) and PB.BOX_PROJECTS <= set(PB.TASKS)
for call in (lambda: PB._lab("amg_dense"), lambda: C.build("cub-10", tile=64)):
    try:
        call()
    except RuntimeError as e:
        assert "lab_sets" in str(e), e
    else:
        raise SystemExit("a lab-only source was accepted")
print("ok")
"""
    r = _run_in_copy(tmp_path, ["backends/pmt/lab_sets.py", "data/sources"], code)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-1500:]


def test_lab_sets_merge_into_backend_tables_when_present():
    from agent_edit.backends.pmt import backend as PB

    L = PB.LAB
    if L is None:
        pytest.skip("lab_sets.py is not in this tree")
    assert PB.SOURCES == L.SOURCES and set(L.COCO_ROWS) <= set(PB.COCO_ROWS)
    assert set(L.TASKS) <= set(PB.TASKS) and L.BOX_PROJECTS <= PB.BOX_PROJECTS
    assert not set(L.TASKS) & {"cub-10", "pets-10", "xview-ships-10"}  # public sets stay in backend.py
    b = L.blank_head(3)
    assert b["class_rows.weight"].shape == (4, 1024) and b["class_rows.bias"].tolist() == [0, 0, 0, 10.0]


def test_image_reads_go_through_the_data_loader():
    from agent_edit import data
    from agent_edit.data import loader

    assert data.open_rgb is loader.open_rgb
    for f in ("backends/pmt/cache.py", "backends/pmt/lab_sets.py", "harness/overlays.py", "baselines/decoder_ft.py"):
        p = PKG / f
        if p.exists():
            assert "data.sources" not in p.read_text(), f


def test_ceiling_is_none_without_a_file(tmp_path, monkeypatch):
    from agent_edit.backends.pmt import evaluate

    assert evaluate.ceiling("cub-10") is None  # AGENT_EDIT_RESULTS unset
    monkeypatch.setenv("AGENT_EDIT_RESULTS", str(tmp_path))
    assert evaluate.ceiling("cub-10") is None  # set, no pmt_a2 file
    (tmp_path / "pmt_a2").mkdir()
    (tmp_path / "pmt_a2" / "cub-10__pmt__eval.json").write_text(json.dumps({"ceiling": 0.75}))
    assert evaluate.ceiling("cub-10") == 0.75


def _tiny_coco(tmp_path, ids):
    imgs = tmp_path / "val2017"
    imgs.mkdir()
    rows = []
    for i in ids:
        Image.fromarray(np.full((6, 8, 3), i, np.uint8)).save(imgs / f"{i:012d}.jpg")
        rows.append({"id": i, "file_name": f"{i:012d}.jpg", "height": 6, "width": 8})
    ann = tmp_path / "instances_val2017.json"
    ann.write_text(json.dumps({"images": rows, "annotations": [], "categories": [{"id": 1, "name": "x"}]}))
    return ann, imgs


def test_guard_cache_builder_writes_the_first_sorted_ids(tmp_path, monkeypatch):
    from agent_edit import data
    from agent_edit.analysis import retention
    from agent_edit.backends.pmt import cache as C
    from agent_edit.backends.pmt import head as H

    ann, imgs = _tiny_coco(tmp_path, [30, 7, 12])
    seen = []

    class FakeBackbone:
        def __init__(self, device):
            assert device == "cpu"

        def features(self, img):
            return img, None

    monkeypatch.setattr(H, "Backbone", FakeBackbone)
    monkeypatch.setattr(H, "build_decoder", lambda device: None)
    monkeypatch.setattr(H, "load_head_weights", lambda device: {})
    monkeypatch.setattr(C, "encode", lambda dec, w, f, rope, hw: {"v": C.VERSION, "hw": tuple(hw), "mean": float(f.mean())})
    real = data.open_rgb
    monkeypatch.setattr(data, "open_rgb", lambda p: seen.append(Path(p).name) or real(p))

    out = tmp_path / "cache"
    done = retention.build_cache(limit=2, root=out, device="cpu", ann=ann, images=imgs)
    assert done == [7, 12] and sorted(p.name for p in out.iterdir()) == ["12.pt", "7.pt"]
    assert seen == ["000000000007.jpg", "000000000012.jpg"]  # read with the data loader
    assert C.load(out, 12) == {"v": C.VERSION, "hw": (6, 8), "mean": 12.0}

    assert retention.build_cache(limit=2, root=out, device="cpu", ann=ann, images=imgs) == []  # resumes
    assert retention.build_cache(limit=None, root=out, device="cpu", ann=ann, images=imgs) == [30]


def test_guard_root_is_the_cache_folder_the_reader_uses(tmp_path, monkeypatch):
    from agent_edit.analysis import retention

    monkeypatch.setenv("AGENT_EDIT_PMT_ROOT", str(tmp_path))
    assert retention.guard_root() == tmp_path / "cache" / "coco_val2017"


def test_missing_pmt_source_names_the_fetch_tool(tmp_path, monkeypatch):
    from agent_edit.backends.pmt import head as H

    monkeypatch.setenv("AGENT_EDIT_PMT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_pmt"):
        H._repo()
    with pytest.raises(FileNotFoundError, match="fetch_pmt"):
        H._checkpoint()
