"""Re-score a finished run and compare with the eval.json it already holds.

Needs the run folders, so it is opt-in:  pytest -m slow tests/test_parity.py
PMT scoring is CPU and must match exactly. The stored `train` block is not compared: the run
scored train images per-class sets do not show, so it differs by design; eval is unaffected.
"""

import json
import os

import pytest

PMT_RUNS = ["stanford-cars-10/pmt_s0_N10_B25_sol"]
EVAL_KEYS = ("macro_f1", "ref_macro_f1", "counts", "mask_ap", "macro_f1_ci95", "bootstrap_minus_ref", "per_class")

pytestmark = pytest.mark.slow


def _run_dir(name):
    if not os.environ.get("AGENT_EDIT_RUNS"):
        pytest.skip("needs AGENT_EDIT_RUNS and the other AGENT_EDIT_* paths")
    from agent_edit import config

    d = config.resolve_run(name)
    if not (d / "eval.json").exists():
        pytest.skip(f"no eval.json in {d}")
    return d


@pytest.mark.parametrize("name", PMT_RUNS)
def test_pmt_rescore_is_exact(name, tmp_path):
    pytest.importorskip("transformers")
    from agent_edit.backends.pmt import evaluate

    d = _run_dir(name)
    old = json.loads((d / "eval.json").read_text())
    new = evaluate.score_run(d, tmp_path / "eval.json")
    for k in EVAL_KEYS:
        assert new["eval"][k] == old["eval"][k], k
