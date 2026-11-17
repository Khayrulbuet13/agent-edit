"""End to end on the toy backend: serve in a subprocess, drive it through the client."""

import json
import os
import subprocess
import sys
import time

import pytest

from agent_edit.harness import client
from agent_edit.harness.client import ClientError


@pytest.fixture
def server(tmp_path):
    run = tmp_path / "run"
    env = {**os.environ, "AGENT_EDIT_HARNESS_HB_S": "0.5"}
    p = subprocess.Popen(
        [sys.executable, "-m", "agent_edit.harness", "serve", "--run-dir", str(run), "--budget", "3", "--max-exec", "4",
         "--backend", "agent_edit.backends.fake:FakeBackend", "--project", "toy", "--max-images", "64"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while not (run / "serve.lock").exists() or not (run / "state.json").exists():
        if p.poll() is not None or time.time() - t0 > 60:
            raise RuntimeError(p.stdout.read())
        time.sleep(0.1)
    yield run, p
    if p.poll() is None:
        p.kill()


def ask(run, req):
    return client.send(run, req, timeout=60)


def test_serve_exec_fwd_final(server):
    run, proc = server
    assert (run / "AGENT_PROMPT.md").read_text().startswith("# TASK")

    r = ask(run, {"kind": "exec", "code": "w = W['fc1.weight'].clone(); w[0] *= -1; save_edit('fc1.weight', w)"})
    assert r["ok"], r["text"]
    assert "[runs 1/4 | passes 0/3]" in r["text"]

    r = ask(run, {"kind": "exec", "code": "import os"})
    assert not r["ok"] and r["text"].startswith("REJECTED")

    r = ask(run, {"kind": "fwd", "request": {"type": "generate", "model": "both", "n_images": 32}})
    assert r["ok"], r["text"]
    res = json.loads((run / "work" / "fwd_1" / "result.json").read_text())
    assert set(res) == {"original", "edited", "paired"}
    assert res["paired"]["n_images"] == 32

    r = ask(run, {"kind": "fwd", "request": {"type": "read", "model": "edited", "n_images": 20}})
    assert r["ok"], r["text"]
    res = json.loads((run / "work" / "fwd_2" / "result.json").read_text())
    assert res["hidden"]["shape"] == [20, 16]

    for bad in ({"type": "read", "leak": True}, {"type": "generate", "n_images": 999},
                {"type": "generate", "n_images": 4, "which": "eval"}, {"type": "read", "fail": True}):
        r = ask(run, {"kind": "fwd", "request": bad})
        assert not r["ok"]
        assert "passes 2/3" in r["text"]  # refused and failed requests are not counted

    r = ask(run, {"kind": "final", "summary": "flipped one unit"})
    assert r["ok"]
    assert proc.wait(timeout=30) == 0

    st = json.loads((run / "state.json").read_text())
    assert st["finished"] and st["passes_used"] == 2 and st["exec_used"] == 1
    man = json.loads((run / "edit" / "manifest.json").read_text())
    assert man["final"] and list(man["edited"]) == ["fc1.weight"] and man["changed_weights"] == 4
    assert "flipped one unit" in (run / "final_answer.md").read_text()
    kinds = [json.loads(line)["kind"] for line in (run / "log" / "turns.jsonl").read_text().splitlines()]
    assert kinds[0] == "serve_start" and "exec_rejected" in kinds and kinds[-1] == "final"

    with pytest.raises(ClientError, match="finished"):
        ask(run, {"kind": "exec", "code": "1"})


def test_restart_keeps_counts_and_edits(server, tmp_path):
    run, proc = server
    ask(run, {"kind": "exec", "code": "b = W['fc1.bias'].clone(); b[0] = 1.0; save_edit('fc1.bias', b)"})
    proc.kill()
    proc.wait()
    (run / "serve.lock").unlink()
    p2 = subprocess.Popen([sys.executable, "-m", "agent_edit.harness", "serve", "--run-dir", str(run), "--budget", "3",
                           "--max-exec", "4", "--backend", "agent_edit.backends.fake:FakeBackend"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not (run / "serve.lock").exists():
            assert time.time() - t0 < 60
            time.sleep(0.1)
        r = ask(run, {"kind": "exec", "code": "edited()"})
        assert "server was restarted" in r["text"] and "'fc1.bias': 1" in r["text"]
        assert "[runs 2/4" in r["text"]
    finally:
        p2.kill()


def test_client_refuses_when_no_server(tmp_path):
    (tmp_path / "queue").mkdir()
    with pytest.raises(ClientError, match="not running"):
        ask(tmp_path, {"kind": "ping"})
