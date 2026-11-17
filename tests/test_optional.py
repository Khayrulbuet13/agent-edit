"""A tree may lack some modules and carry the package under another name; the code must cope."""

import argparse
import importlib
import sys
import types

import pytest
import torch

from agent_edit import cli, config
from agent_edit.analysis import replay
from agent_edit.baselines import run as baselines_run
from agent_edit.harness import server


def _present(monkeypatch, mods):
    """Make exactly `mods` count as present in the registry and the other has_module users."""
    fake = lambda rel: rel in mods  # noqa: E731
    for owner in (cli, replay, server, config):
        monkeypatch.setattr(owner, "has_module", fake)


def test_has_module_reads_the_disk_and_imports_no_parent(tmp_path, monkeypatch):
    pkg = tmp_path / "optpkg"
    (pkg / "a").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a" / "__init__.py").write_text("raise RuntimeError('parent imported')")
    (pkg / "a" / "b.py").write_text("")
    (pkg / "empty_dir").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(config, "PKG", "optpkg")
    assert config.has_module("a.b")
    assert config.has_module("a")
    assert not config.has_module("a.missing")
    assert not config.has_module("missing.b")
    assert not config.has_module("empty_dir")  # a bare folder is not a module
    assert "optpkg.a" not in sys.modules


def test_absent_modules_drop_their_commands(monkeypatch, capsys):
    _present(monkeypatch, {"backends.pmt.evaluate", "analysis.replay", "baselines.probe", "baselines.adam"})
    assert [c.name for c in cli.active()] == ["score", "baselines", "replay", "env"]
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "replay" in out and "baselines" in out
    for gone in ("formula", "retention", "cache"):
        assert gone not in out


def test_choices_come_from_what_is_present(monkeypatch):
    _present(monkeypatch, {"baselines.probe", "baselines.adam", "backends.pmt.evaluate", "backends.pmt.cache"})

    def choices(add, cmd, dest):
        sub = argparse.ArgumentParser().add_subparsers()
        add(sub)
        return next(a for a in sub.choices[cmd]._actions if a.dest == dest).choices

    assert choices(cli._add_baselines, "baselines", "method") == ["c2", "c5r"]
    assert choices(cli._add_score, "score", "model") == ["auto", "pmt"]
    assert choices(cli._add_cache, "cache", "model") == ["pmt"]
    with pytest.raises(SystemExit):
        cli.main(["baselines", "c5f", "cub-10"])  # its module is absent




def test_cv_kinds_follow_the_modules_present(monkeypatch):
    monkeypatch.setattr(config, "has_module", lambda rel: rel in {"baselines.probe", "baselines.decoder_ft"})
    try:
        partial = importlib.reload(baselines_run)
        assert list(partial.FITS) == ["C2"]
        assert partial.CV_METHODS == ("c2", "c5f", "lwf")
        with pytest.raises(SystemExit, match="no cv version"):
            partial.main("c5r", "cub-10")
    finally:
        monkeypatch.undo()
        importlib.reload(baselines_run)


def test_formula_forwards_its_arguments(monkeypatch):
    got = []
    stub = types.ModuleType("agent_edit.baselines.formula")
    stub.main = lambda argv: got.append(argv) or 0
    monkeypatch.setitem(sys.modules, "agent_edit.baselines.formula", stub)
    _present(monkeypatch, {"baselines.formula"})
    assert cli.main(["formula", "cub-10", "--n", "5", "--seed", "2", "--out", "/o"]) == 0
    assert got == [["cub-10", "--n", "5", "--seed", "2", "--out", "/o"]]
    with pytest.raises(SystemExit) as e:
        cli.main(["formula", "cub-10"])
    assert e.value.code == 2 and len(got) == 1


def test_command_missing_its_module_says_so(monkeypatch):
    _present(monkeypatch, set())
    with pytest.raises(SystemExit, match="no baselines.formula"):
        cli._run_formula(argparse.Namespace(project="p", n=None, seed=None, out=None))




def test_stored_backend_path_is_resolved_against_this_package(monkeypatch):
    monkeypatch.setattr(replay, "PKG", "agent_edit")
    monkeypatch.setattr(replay, "has_module", lambda rel: rel in {"backends.pmt.backend", "backends.fake"})
    assert replay.own_spec("agent_edit.backends.pmt.backend:make_backend") == "agent_edit.backends.pmt.backend:make_backend"
    assert replay.own_spec("agent_edit.backends.fake:FakeBackend") == "agent_edit.backends.fake:FakeBackend"
    assert replay.own_spec("elsewhere.my_backend:make") == "elsewhere.my_backend:make"
    assert replay.own_spec("plainmodule:make") == "plainmodule:make"


def test_backend_tail_ignores_the_root():
    for root in ("agent_edit", "agent_edit"):
        spec = f"{root}.backends.pmt.backend:make_backend"
        assert replay._tail(spec) == "backends.pmt.backend"
        assert replay._scorer(spec) is replay._eval_pmt
    assert replay._under("backends.pmt", "backends.pmt")
    assert not replay._under("backends.pmt_other", "backends.pmt")
    with pytest.raises(SystemExit, match="no eval scorer"):
        replay._scorer("agent_edit.backends.unknown.backend:make")


def test_parent_row_fix_still_acts_under_another_root(monkeypatch):
    coco_w = torch.arange(80 * 4, dtype=torch.float32).reshape(80, 4)
    monkeypatch.setattr("agent_edit.backends.pmt.head.load_head_weights",
                        lambda device="cpu": {"class_head.weight": coco_w, "class_head.bias": torch.arange(80.0)})
    w = {"class_rows.weight": torch.zeros(11, 4), "class_rows.bias": torch.zeros(11)}
    for root in ("agent_edit", "agent_edit"):
        out = replay._fg_parent_fix(f"{root}.backends.pmt.backend:make_backend", "stanford-cars-10", "cub-10", w)
        assert out is not w and torch.equal(out["class_rows.weight"][2], coco_w[14])


def test_clip_scorer_is_listed_only_when_present(monkeypatch):
    assert ("backends.clip" in replay.SCORERS) == config.has_module("backends.clip.evaluate")


def test_bridge_variables_are_listed_only_with_the_bridge(monkeypatch):
    stub = types.ModuleType(f"{config.PKG}.{config.BRIDGE_FILE}")
    stub.ENV = {"AGENT_EDIT_BRIDGE_ROOT": "the bridge's checkout"}
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(config, "has_module", lambda rel: False)
    assert "AGENT_EDIT_BRIDGE_ROOT" not in config.check()
    monkeypatch.setattr(config, "has_module", lambda rel: rel == config.BRIDGE_FILE)
    assert "AGENT_EDIT_BRIDGE_ROOT" in config.check()
    monkeypatch.delenv("AGENT_EDIT_BRIDGE_ROOT", raising=False)
    with pytest.raises(config.ConfigError, match="AGENT_EDIT_BRIDGE_ROOT is not set. Point it to the bridge's checkout"):
        config.env_path("AGENT_EDIT_BRIDGE_ROOT")


def test_command_name_is_the_package_name_with_dashes():
    assert config.cli_name("some_pkg") == "some-pkg"
    assert config.cli_name("pkg") == "pkg"


def test_reading_mode_without_its_module_gives_a_clear_error(monkeypatch):
    _present(monkeypatch, set())
    with pytest.raises(SystemExit, match="--reading needs"):
        server._reading_module()


def test_package_name_comes_from_the_package():
    from agent_edit.harness import guards
    from agent_edit.metrics import docs

    assert config.PKG in guards.DENY_MODULES
    assert docs.make_doc("p", "k", (1, 1), [])["model"] == config.PKG




def test_wrapper_names_the_command_and_the_module(tmp_path, monkeypatch):
    from agent_edit.harness import prompt as P

    py = tmp_path / "bin" / "python"
    py.parent.mkdir()
    (py.parent / "agent-edit").write_text("")
    run = tmp_path / "run"
    run.mkdir()
    (run / "AGENT_PROMPT.md").write_text(f"x `{py} -m agent_edit.harness status {run}` y")
    monkeypatch.setattr(P, "PKG", "agent_edit")
    monkeypatch.setattr(P, "CLI", "agent-edit")
    text = P.render_prompt(run, "proj", 5, 10)
    assert f"{py.parent}/agent-edit status {run}" in text
    assert "-m agent_edit.harness" not in text


def test_prompt_wrapped_prints_the_full_text(tmp_path, capsys):
    from agent_edit.cli import main

    args = ["prompt", "--project", "toy", "--budget", "5", "--backend", "agent_edit.backends.fake:FakeBackend",
            "--run-dir", str(tmp_path / "r")]
    assert main([*args, "--wrapped"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("You are the EDITING AGENT") and 'The project is "toy"' in out
    assert main([*args, "--wrapped", str(tmp_path / "full.md")]) == 0
    assert (tmp_path / "full.md").read_text() + "\n" == out
    assert main([*args]) == 0 and capsys.readouterr().out.strip().endswith("AGENT_PROMPT.md")
    assert main(["prompt", "--project", "toy", "--budget", "5", "--backend", "agent_edit.backends.fake:FakeBackend",
                 "--wrapped"]) == 2
