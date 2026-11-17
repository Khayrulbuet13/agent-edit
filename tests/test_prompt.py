import json
from pathlib import Path

import pytest

from agent_edit.harness.prompt import build_prompt

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("case", ["plain", "flags"])
def test_prompt_matches_research_code(case):
    """prompt_<case>.md pins the text the agent reads; it must not drift."""
    kw = json.loads((DATA / f"prompt_{case}.json").read_text())
    assert build_prompt(**kw) == (DATA / f"prompt_{case}.md").read_text()


def test_metric_wording_comes_from_the_backend():
    kw = json.loads((DATA / "prompt_plain.json").read_text())
    text = build_prompt(**kw, metric="top-1 accuracy", metric_how="The score is top-1 accuracy: the share of images right.")
    assert "as measured by top-1 accuracy on held-out images." in text
    assert "The score is top-1 accuracy: the share of images right." in text and "IoU 0.5" not in text


def test_toy_backend_brings_its_own_intro_and_the_default_stays():
    from agent_edit.backends.fake import FakeBackend

    kw = json.loads((DATA / "prompt_plain.json").read_text())
    assert kw["model_intro"] == FakeBackend().project_info()["model_intro"]
    assert "a toy network" in (DATA / "prompt_plain.md").read_text()
    default = build_prompt(**{k: v for k, v in kw.items() if k != "model_intro"})
    assert "the head of a vision model" in default
