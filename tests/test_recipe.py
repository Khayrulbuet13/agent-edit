"""--recipe: the kernel's `train`, check_recipe(), the reproduce check at final, and replay."""

import json

import pytest

from agent_edit.analysis.replay import replay
from agent_edit.cli import main
from conftest import ask, fwd

SPEC = '{"type": "read", "model": "original", "n_images": 64}'
# the agent's edit, worked out in the kernel from a fwd read it paid for
EDIT = """
import json
res = json.load(open(WORK + '/fwd_1/result.json'))
r, y = np.load(res['hidden']['npy']), np.load(res['matched_class']['npy'])
w = W['fc2.weight'].clone()
for c in range(3):
    w[c] += 0.5 * torch.from_numpy(r[y == c].mean(0))
save_edit('fc2.weight', w)
"""
RECIPE = """
import json


def edit(train, weights):
    r = train.read(json.loads('SPEC'))
    h, y = torch.from_numpy(r["hidden"]), torch.from_numpy(r["matched_class"])
    w = weights["fc2.weight"].clone()
    for c in range(3):
        w[c] += STRENGTH * h[y == c].mean(0)
    return {"fc2.weight": w}
""".replace("SPEC", SPEC)


def write_recipe(srv, text):
    ok, out, _ = ask(srv, "exec", code=f"open(WORK + '/recipe.py', 'w').write({text!r})")
    assert ok, out


@pytest.fixture
def run(make_server):
    srv = make_server(recipe=True)
    ok, text, _ = fwd(srv, **json.loads(SPEC))
    assert ok, text
    ok, text, _ = ask(srv, "exec", code=EDIT)
    assert ok, text
    return srv


def test_recipe_that_reproduces(run):
    write_recipe(run, RECIPE.replace("STRENGTH", "0.5"))
    ok, text, _ = ask(run, "exec", code="check_recipe()")
    assert ok and "reproduces the saved edits" in text, text
    ok, text, done = ask(run, "final", summary="class means")
    assert ok and done and "reproduces" in text
    fin = json.loads((run.run / "final.json").read_text())
    assert fin["recipe"]["reproduces"] and fin["recipe"]["max_abs_diff"] <= 1e-5
    assert (run.run / "recipe.py").read_text() == (run.run / "work" / "recipe.py").read_text()


def test_recipe_that_does_not_reproduce_never_blocks_final(run):
    write_recipe(run, RECIPE.replace("STRENGTH", "0.25"))
    ok, text, _ = ask(run, "exec", code="check_recipe()")
    assert ok and "does NOT reproduce" in text
    ok, _, done = ask(run, "final", summary="x")
    assert ok and done
    fin = json.loads((run.run / "final.json").read_text())
    assert fin["recipe"]["present"] and not fin["recipe"]["reproduces"] and fin["recipe"]["max_abs_diff"] > 1e-5


def test_missing_recipe_is_recorded(run):
    ok, text, done = ask(run, "final", summary="x")
    assert ok and done and "no recipe" in text
    assert json.loads((run.run / "final.json").read_text())["recipe"] == {"present": False, "reproduces": False}


@pytest.mark.parametrize("body", [
    "    return {'fc2.weight': torch.from_numpy(np.load(PATH)).float()}",   # reads a file the agent saved
    "    open('/etc/hostname').read()\n    return {}",
])
def test_recipe_may_not_read_files(run, body):
    ok, _, _ = ask(run, "exec", code="np.save(WORK + '/w.npy', W['fc2.weight'].numpy())")
    assert ok
    path = repr(str(run.kernel.work / "w.npy"))
    write_recipe(run, "P = " + path + "\nPATH = P\n\ndef edit(train, weights):\n" + body + "\n")
    ok, text, _ = ask(run, "exec", code="check_recipe()")
    assert ok and "failed" in text and ("not allowed" in text or "only works inside" in text), text


def test_train_serves_only_reads_already_paid_for(run):
    ok, text, _ = ask(run, "exec", code="train.read({'n_images': 8})")
    assert not ok and "Run it with the fwd tool first" in text
    ok, text, _ = ask(run, "exec", code=f"r = train.read({SPEC}); r['hidden'].shape")
    assert ok and "(64, 16)" in text, text
    ok, text, _ = ask(run, "exec", code="w = {'fc2.weight': W['fc2.weight']}; train.read({'n_images': 64}, weights=w)")
    assert not ok and "LookupError" in text  # read on the edited head was never run


def test_replay_on_own_pool_and_another(run):
    write_recipe(run, RECIPE.replace("STRENGTH", "0.5"))
    assert ask(run, "final", summary="x")[2]
    own = replay(run.run)
    assert own["ok"] and own["same_pool"] and own["matches_saved_edit"]["reproduces"]
    assert own["eval"]["n_images"] == 256 and own["edited_tensors"] == ["fc2.weight"]
    other = replay(run.run, seed=3, n=128)
    assert other["ok"] and not other["same_pool"] and "matches_saved_edit" not in other
    assert other["train"]["n_images"] == 128
    assert json.loads((run.run / "replay" / "toy_s3_N128.json").read_text())["eval"]["macro_f1"] == other["eval"]["macro_f1"]


def test_replay_cli(run):
    write_recipe(run, RECIPE.replace("STRENGTH", "0.5"))
    ask(run, "final", summary="x")
    assert main(["replay", str(run.run), "--seed", "1", "--n", "64", "--no-eval"]) == 0
    rec = json.loads((run.run / "replay" / "toy_s1_N64.json").read_text())
    assert rec["ok"] and "eval" not in rec


def test_no_flag_no_train_object(make_server):
    srv = make_server()
    ok, text, _ = ask(srv, "exec", code="train")
    assert not ok and "NameError" in text
    assert "recipe.py" not in (srv.run / "AGENT_PROMPT.md").read_text()


def test_live_reads_of_the_whole_pool_follow_the_target(tmp_path):
    from agent_edit.backends.fake import FakeBackend
    from agent_edit.harness.recipe import LiveTrain

    be = FakeBackend(seed=1, n_shot=128)
    train = LiveTrain(be, be.initial_weights(), tmp_path, all_images=(64, 128))
    assert train.read({"n_images": 64})["hidden"].shape == (128, 16)
    assert train.read({"n_images": 32})["hidden"].shape == (32, 16)
