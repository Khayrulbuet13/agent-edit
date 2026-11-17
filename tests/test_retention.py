"""--retention: fwd {"which": "retain"} scores the original task for one pass; a prompt rule asks to keep it."""

import json

import pytest

from agent_edit.backends.fake import FakeBackend
from conftest import ask, fwd, result

RULE = "keep retention within 1 point of the original"


def test_retain_costs_one_pass_and_reports_both_heads(make_server):
    srv = make_server(retention=True)
    ok, text, _ = fwd(srv, which="retain")
    assert ok, text
    assert "pass 1/6" in text
    res = result(srv, 1)
    assert res["original"] == res["edited"] == 100.0 and res["within_1_point"]

    ask(srv, "exec", code="w = W['fc2.weight'].clone(); w[0] += 50; save_edit('fc2.weight', w)")
    assert fwd(srv, which="retain")[0]
    res = result(srv, 2)
    assert res["edited"] < 99 and res["difference"] < -1 and not res["within_1_point"]
    assert srv.passes == 2


def test_retain_refused_without_the_flag(make_server):
    srv = make_server()
    ok, text, _ = fwd(srv, which="retain")
    assert not ok and "not counted" in text and srv.passes == 0


def test_rule_in_prompt_only_with_the_flag(make_server):
    on = (make_server(name="on", retention=True).run / "AGENT_PROMPT.md").read_text()
    off = (make_server(name="off").run / "AGENT_PROMPT.md").read_text()
    assert RULE in on and '{"which": "retain"}' in on and "64 held-out points" in on
    assert RULE not in off and "retain" not in off


def test_backend_without_retention_is_refused(make_server):
    class NoRetention(FakeBackend):
        def retention(self, weights):
            return None

    with pytest.raises(SystemExit, match="retention"):
        make_server(NoRetention(), retention=True)


def test_final_records_retention(make_server):
    srv = make_server(retention=True)
    ask(srv, "exec", code="w = W['fc2.weight'].clone(); w[1] -= 50; save_edit('fc2.weight', w)")
    ok, _, done = ask(srv, "final", summary="x")
    assert ok and done
    fin = json.loads((srv.run / "final.json").read_text())
    assert fin["flags"]["retention"] and fin["retention"]["edited"] < 100


def test_flags_are_pinned_across_restarts(make_server):
    srv = make_server(retention=True)
    srv._release_lock()
    with pytest.raises(SystemExit, match="flags"):
        make_server(retention=False)
