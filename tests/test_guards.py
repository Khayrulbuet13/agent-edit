from pathlib import Path

import pytest

from agent_edit.harness.guards import bad_module, check_keys, inside, precheck

WORK = Path("/tmp/guards-test-work")


@pytest.mark.parametrize("code", [
    "import os",
    "from subprocess import run",
    "import torch.optim as o",
    "from torch import autograd",
    "from torch.func import grad",
    "x = torch.zeros(3, requires_grad=True)",
    "w.requires_grad = True",
    "loss.backward()",
    "w.requires_grad_()",
    "getattr(torch, 'optim')",
    "eval('1')",
    "__import__('os')",
    "head_forward(f)",
    "import sam3",
    "import agent_edit",
    "open('/etc/passwd')",
    "np.load('rel.npy')",
    "torch.load('/elsewhere/t.pt')",
    "torch.set_grad_enabled(True)",
    "torch.save(t, '/tmp/t.pt')",
    "torch.save(t, f='/tmp/t.pt')",
    "torch.save(t, 'rel.pt')",
    f"torch.save(t, '{WORK}/../t.pt')",
    "np.savez('/tmp/a.npz', a=x)",
    "np.savez_compressed(file='/tmp/a.npz', a=x)",
    "numpy.savetxt('/tmp/a.txt', x, fmt='%.3f')",
    "np.genfromtxt('/data/eval.csv')",
    "x.tofile('/tmp/a.bin')",
    "plt.savefig('/tmp/a.png', dpi=80)",
    "df.to_csv(path_or_buf='/tmp/a.csv')",
    "open(WORK + '/a', 'w', newline='/tmp/x')",
    f"np.save('{WORK}/a.npy', np.load('/tmp/other.npy'))",
])
def test_precheck_refuses(code):
    assert precheck(code, WORK) is not None


@pytest.mark.parametrize("code", [
    "import numpy as np\nx = np.arange(3).mean()",
    "w = W['fc1.weight'].clone(); w[0] += 1; save_edit('fc1.weight', w)",
    "x = torch.zeros(3, requires_grad=False)",
    f"np.save('{WORK}/a.npy', np.zeros(2))",
    "open('notes.txt', 'w')",
    f"torch.save(t, '{WORK}/t.pt')",
    f"np.savetxt('{WORK}/a.txt', x, fmt='%.3f', delimiter=',')",
    f"x.tofile('{WORK}/a.bin')",
    "torch.save(t, WORK + '/t.pt')",
    "s.split('/')",
    "import math",
    "def broken(:",  # syntax errors pass the pre-check; the server reports them
])
def test_precheck_allows(code):
    assert precheck(code, WORK) is None


def test_precheck_names_the_line():
    assert precheck("x = 1\nimport os", WORK).startswith("line 2:")


def test_bad_module():
    assert bad_module("numpy") is None
    assert bad_module("torch.nn") is None
    assert "torch.optim" in bad_module("torch.optim.adam")
    assert bad_module("transformers") is not None


def test_inside(tmp_path):
    assert inside(str(tmp_path / "a" / "b"), tmp_path, relative_ok=False)
    assert inside("a/b", tmp_path, relative_ok=True)
    assert not inside("a/b", tmp_path, relative_ok=False)
    assert not inside(str(tmp_path / ".." / "x"), tmp_path, relative_ok=False)


@pytest.mark.parametrize("res", [{"gt_masks": 1}, {"a": {"eval_f1": 1}}, {"rows": [{"raw_gt": 0}]}, {"eval": 1}])
def test_check_keys_refuses_eval_and_gt(res):
    with pytest.raises(RuntimeError):
        check_keys(res)


def test_check_keys_allows_train_results():
    check_keys({"macro_f1": 0.5, "per_class": {"car": {"f1": 1.0}}, "matched_class": [1, 2], "evaluation_note": "x"})


def run_in_kernel(tmp_path, code):
    from agent_edit.backends.fake import FakeBackend
    from agent_edit.harness.files import ensure_dirs
    from agent_edit.harness.kernel import Kernel

    ensure_dirs(tmp_path)
    k = Kernel(tmp_path, FakeBackend().initial_weights(), [], lambda r: None)
    return k, *k.run_code(code, timeout=60, max_chars=10_000)


@pytest.mark.parametrize("code", [
    "p = OUT + '/t.pt'; torch.save(W['fc1.bias'], p)",
    "import torch.serialization as s; p = OUT + '/t.pt'; s.save(W['fc1.bias'], p)",
    "p = OUT + '/a.npy'; np.save(p, np.zeros(2))",
    "p = OUT + '/a.bin'; np.zeros(2).tofile(p)",
    "p = OUT + '/a.txt'; np.savetxt(p, np.zeros(2))",
])
def test_runtime_refuses_writes_outside_work(tmp_path, code):
    out = tmp_path / "outside"
    out.mkdir()
    _, text, ok = run_in_kernel(tmp_path / "run", f"OUT = {str(out)!r}\n" + code)
    assert not ok and any(m in text for m in ("not allowed", "must be inside", "only works inside")), text
    assert not any(out.iterdir())


def test_runtime_allows_work_and_save_edit(tmp_path):
    import torch

    k, text, ok = run_in_kernel(tmp_path, "p = WORK + '/t.pt'; torch.save(W['fc1.bias'], p); np.save(WORK + '/a.npy', "
                                          "np.ones(2)); b = load('fc1.bias'); b[0] = 3.0; save_edit('fc1.bias', b)")
    assert ok, text
    assert (tmp_path / "work" / "t.pt").exists() and (tmp_path / "edit" / "fc1.bias.pt").exists()
    assert not hasattr(torch.save, "__wrapped__")  # unwrapped once agent code is done
    torch.save(torch.zeros(1), tmp_path / "server_side.pt")  # the server itself is not confined
