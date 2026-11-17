"""--seed-recipe PATH: a warm start copied from another project's recipe.py, not an agent capability
flag, so it is not in Server.flags, but its source path is still pinned in harness.json the same way."""

import json

import pytest


def _source_recipe(tmp_path, project="cars-10", run="pmt_s0_N20_agent"):
    d = tmp_path / "src_runs" / project / run
    d.mkdir(parents=True)
    p = d / "recipe.py"
    p.write_text("def edit(train, weights):\n    return {}\n")
    return p


def test_seed_recipe_copied_into_work(make_server, tmp_path):
    src = _source_recipe(tmp_path)
    srv = make_server(seed_recipe=str(src))
    dst = srv.run / "work" / "seed_recipe.py"
    assert dst.is_file()
    assert dst.read_text() == src.read_text()


def test_seed_recipe_source_pinned_in_harness_json(make_server, tmp_path):
    src = _source_recipe(tmp_path)
    srv = make_server(seed_recipe=str(src))
    hp = json.loads((srv.run / "harness.json").read_text())
    assert hp["seed_recipe_source"] == str(src.resolve())


def test_seed_recipe_label_defaults_to_grandparent_folder(make_server, tmp_path):
    src = _source_recipe(tmp_path, project="cars-10", run="pmt_s0_N20_agent")
    srv = make_server(seed_recipe=str(src))
    text = (srv.run / "AGENT_PROMPT.md").read_text()
    assert "another dataset (cars-10)" in text


def test_seed_recipe_label_override(make_server, tmp_path):
    src = _source_recipe(tmp_path)
    srv = make_server(seed_recipe=str(src), seed_recipe_source="stanford-cars-10 (pilot)")
    text = (srv.run / "AGENT_PROMPT.md").read_text()
    assert "another dataset (stanford-cars-10 (pilot))" in text


def test_seed_recipe_prompt_paragraph_only_with_the_flag(make_server, tmp_path):
    src = _source_recipe(tmp_path, project="cars-10")
    on = (make_server(name="on", seed_recipe=str(src)).run / "AGENT_PROMPT.md").read_text()
    off = (make_server(name="off").run / "AGENT_PROMPT.md").read_text()
    assert "seed_recipe.py` is a recipe that worked on another dataset (cars-10)" in on
    assert "seed_recipe.py" not in off


def test_seed_recipe_missing_file_refused(make_server, tmp_path):
    with pytest.raises(SystemExit, match="seed-recipe"):
        make_server(seed_recipe=str(tmp_path / "nope.py"))


def test_seed_recipe_resume_with_a_different_source_refused(make_server, tmp_path):
    src1 = _source_recipe(tmp_path, project="a", run="r1")
    src2 = _source_recipe(tmp_path, project="b", run="r2")
    srv = make_server(seed_recipe=str(src1))
    srv._release_lock()
    with pytest.raises(SystemExit, match="seed_recipe_source"):
        make_server(seed_recipe=str(src2))


def test_no_seed_recipe_no_change(make_server):
    """Off by default: no file, no harness.json key, no prompt text -- byte-identical to a run without the flag."""
    srv = make_server()
    assert not (srv.run / "work" / "seed_recipe.py").exists()
    hp = json.loads((srv.run / "harness.json").read_text())
    assert "seed_recipe_source" not in hp
