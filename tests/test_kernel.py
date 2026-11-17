import json

import pytest
import torch

from agent_edit.backends.fake import FakeBackend
from agent_edit.harness.files import ensure_dirs
from agent_edit.harness.kernel import EditRejected, Kernel


@pytest.fixture
def kernel(tmp_path):
    ensure_dirs(tmp_path)
    log = []
    orig = {k: v.float() for k, v in FakeBackend().initial_weights().items()}
    k = Kernel(tmp_path, orig, FakeBackend().frozen_patterns(), log.append)
    k.log_records = log
    return k


def test_save_edit_writes_file_and_manifest(kernel, tmp_path):
    w = kernel.W["fc1.weight"].clone()
    w[0, 0] += 1.0
    msg = kernel.save_edit("fc1.weight", w)
    assert "1 of 64 elements changed" in msg
    man = json.loads((tmp_path / "edit" / "manifest.json").read_text())
    assert man["edited"]["fc1.weight"]["changed"] == 1
    assert man["changed_weights"] == 1
    assert torch.equal(torch.load(tmp_path / "edit" / "fc1.weight.pt"), w)
    assert kernel.log_records[-1]["kind"] == "save_edit"


@pytest.mark.parametrize("name,make,why", [
    ("nope", lambda w: w, "unknown tensor"),
    ("fc2.bias", lambda w: w + 1, "frozen"),
    ("fc1.weight", lambda w: w.double(), "dtype"),
    ("fc1.weight", lambda w: w[:2], "shape"),
    ("fc1.weight", lambda w: w * float("nan"), "NaN"),
    ("fc1.weight", lambda w: w.clone(), "identical"),
    ("fc1.weight", lambda w: w * 2, "rescales"),
])
def test_save_edit_refusals(kernel, name, make, why):
    base = kernel.W.get(name, kernel.W["fc1.weight"])
    with pytest.raises(EditRejected, match=why):
        kernel.save_edit(name, make(base.clone()))


def test_saved_edits_survive_restart(kernel, tmp_path):
    w = kernel.W["fc1.bias"].clone()
    w[3] = 2.0
    kernel.save_edit("fc1.bias", w)
    again = Kernel(tmp_path, kernel.orig, [], lambda r: None)
    assert torch.equal(again.W["fc1.bias"], w)
    assert again.edited() == {"fc1.bias": 1}


def test_drop_edit(kernel, tmp_path):
    w = kernel.W["fc1.bias"].clone()
    w[0] = 5.0
    kernel.save_edit("fc1.bias", w)
    kernel.drop_edit("fc1.bias")
    assert not (tmp_path / "edit" / "fc1.bias.pt").exists()
    assert torch.equal(kernel.W["fc1.bias"], kernel.orig["fc1.bias"])


def test_run_code_shows_last_expression_and_keeps_state(kernel):
    out, ok = kernel.run_code("x = 41", 0, 1000)
    assert ok and out == ""
    out, ok = kernel.run_code("x + 1", 0, 1000)
    assert ok and out.strip() == "42"


def test_run_code_reports_errors_without_harness_frames(kernel):
    out, ok = kernel.run_code("1 / 0", 0, 1000)
    assert not ok
    assert "ZeroDivisionError" in out
    assert "kernel.py" not in out


def test_run_code_blocks_imports_and_files_at_runtime(kernel):
    out, ok = kernel.run_code("m = 'o' + 's'\n__builtins__['__import__'](m)", 0, 1000)
    assert not ok
    out, ok = kernel.run_code("open('/etc/hostname')", 0, 1000)
    assert not ok and "PermissionError" in out


def test_run_code_cuts_long_output(kernel):
    out, ok = kernel.run_code("print('x' * 5000)", 0, 100)
    assert ok and "[output cut" in out


def test_originals_are_not_exposed(kernel):
    kernel.run_code("W['fc1.weight'][0, 0] = 99.0", 0, 1000)
    assert kernel.orig["fc1.weight"][0, 0] != 99.0
    out, _ = kernel.run_code("load_original('fc1.weight')[0, 0].item() != 99.0", 0, 1000)
    assert out.strip() == "True"


def test_one_element_tensor_can_change(tmp_path):
    ensure_dirs(tmp_path)
    k = Kernel(tmp_path, {"logit_scale": torch.tensor([100.0]), "w": torch.ones(3)}, [], lambda r: None)
    assert "1 of 1 elements changed" in k.save_edit("logit_scale", torch.tensor([50.0]))
    with pytest.raises(EditRejected, match="identical"):
        k.save_edit("logit_scale", torch.tensor([100.0]))
    with pytest.raises(EditRejected, match="rescales"):
        k.save_edit("w", torch.ones(3) * 2)
