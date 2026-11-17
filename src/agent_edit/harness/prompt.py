"""The agent's task prompt (AGENT_PROMPT.md, filled from prompts/agent.md) and the wrapper around it
that an agent adapter hands to its CLI (prompts/run_template.md)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..config import CLI, PKG

TEMPLATE = Path(__file__).with_name("prompts") / "agent.md"
RUN_TEMPLATE = Path(__file__).with_name("prompts") / "run_template.md"
READING_TEMPLATE = Path(__file__).with_name("prompts") / "reading.md"
DEFAULT_INTRO = "You are given the weights of the head of a vision model."
# The prompt says "work from the folder <repo>"; that is the checkout root when running from source.
REPO_ROOT = Path(__file__).resolve().parents[3]
METRIC = "instance macro-F1"
METRIC_HOW = ("The score is instance macro-F1: per class, predicted instances are matched to true instances at IoU 0.5 "
              "and F1 is computed; then it is averaged over classes.")


def build_prompt(project, budget, max_exec, run, describe_text, classes, task="",
                 exec_timeout=900, max_chars=10_000, max_images=256, py=None, repo=None,
                 n_shot=None, gate_open=False, model_intro=None, overlays=False, overlay_legend="",
                 overlay_sheet=False, retention=None, recipe=False, hide_names=False, seed_recipe_source=None,
                 metric=None, metric_how=None) -> str:
    """`retention` is the backend's sentence on what the retain check scores (None = check off).
    `metric` / `metric_how` name and explain the held-out score (default: detection macro-F1).
    `seed_recipe_source` is the source project's label when the run was served with --seed-recipe,
    else None. Every flag left at its default adds nothing to the prompt."""
    # n_shot and gate_open add text only when set
    shot_note = ""
    if n_shot:
        shot_note = (f" The labelled data is exactly {n_shot} training image{'s' if n_shot != 1 else ''}: "
                     f"these {n_shot} are the only labelled images there are, and the only ones the tools can run on.")
    gate_note = ""
    if gate_open:
        gate_note = ("\nThe presence gate is already open: the head you start from has its presence logit pushed high for "
                     "every image and prompt, so presence is about 1 and every query is scored by its class logit alone. "
                     "The failures that remain are about which instances are the project's objects and classes, "
                     "not about whether the class is present.\n")
    py, run = py or sys.executable, Path(run)
    flag_notes = _flag_notes(py, run, overlays, overlay_legend, overlay_sheet, retention, recipe, hide_names,
                             seed_recipe_source)
    return TEMPLATE.read_text().format(
        intro=model_intro or DEFAULT_INTRO, project=project, task=task, gate_note=gate_note, **flag_notes,
        metric=metric or METRIC, metric_how=metric_how or METRIC_HOW,
        cls=", ".join(classes) if classes else "(see the head description)",
        repo=repo or REPO_ROOT, py=py, run=run, budget=budget, max_exec=max_exec,
        exec_timeout=exec_timeout, max_chars=max_chars, max_images=max_images, n_demo=min(64, max_images), pkg=PKG,
        shot_note=shot_note, describe=describe_text.strip(),
    )


def build_reading_prompt(project, run, n_rows, class_names, max_exec, exec_timeout=900, max_chars=10_000,
                         py=None, repo=None, task="") -> str:
    """The reading-mode prompt (--reading): name the shuffled rows, no head to edit."""
    py, run = py or sys.executable, Path(run)
    return READING_TEMPLATE.read_text().format(
        project=project, task=task, n_rows=n_rows, cls=", ".join(class_names),
        repo=repo or REPO_ROOT, py=py, run=run, max_exec=max_exec, exec_timeout=exec_timeout, max_chars=max_chars,
    )


def _flag_notes(py, run, overlays, overlay_legend, overlay_sheet, retention, recipe, hide_names,
                seed_recipe_source=None) -> dict:
    notes = dict.fromkeys(("names_note", "kernel_note", "overlay_note", "retain_tool", "retain_rule",
                           "recipe_note", "seed_recipe_note"), "")
    if hide_names:
        notes["names_note"] = (" The class names are hidden: each class is called class_<k> everywhere you look, "
                               "and that is all you are told about it.")
    if overlays:
        notes["overlay_note"] = (" Every generate also draws pictures of the errors on the images it scored (PNG files; "
                                 "their paths are under \"overlays\" in the result" + (f"; {overlay_legend}" if overlay_legend else "")
                                 + "). Look at them: they show where the head fails and what it confuses.")
        if overlay_sheet:
            from .overlays import SHEET_K

            notes["overlay_note"] += (f" Each pass also writes a sheet.png in that folder tiling the {SHEET_K} "
                                      "worst-scored images of the pass, worst first, each one labelled with its "
                                      "index, true class and predicted class: open that one first.")
    if seed_recipe_source:
        notes["seed_recipe_note"] = (f" `{run}/work/seed_recipe.py` is a recipe that worked on another dataset "
                                     f"({seed_recipe_source}). It may or may not fit this one; you may start from "
                                     "it, change it, or ignore it.")
    if retention:
        notes["retain_tool"] = (f"\n   `{py} -m {PKG}.harness fwd {run} --json '{{\"which\": \"retain\"}}'`\n"
                                f"     Retention check: {retention} It returns the original head's score, your edited head's "
                                "score and their difference, in points. One pass.")
        notes["retain_rule"] = ("\n(f) Breaking what the model already does: keep retention within 1 point of the original "
                                "(the retain check). Run it on your final edits before you call final.")
    if recipe:
        notes["kernel_note"] = (" With them: `train` (the train data as a recipe sees it; see HOW IT ENDS) and "
                                "`check_recipe()` (runs WORK/recipe.py the way the final check does and says whether it "
                                "gives back your saved edits).")
        notes["recipe_note"] = (
            f"\nYour submission also includes the method, as a program: write `{run}/work/recipe.py` that defines "
            "`edit(train, weights)` and returns a dict tensor name -> new tensor, the same tensors you saved with "
            "save_edit. `weights` holds copies of the ORIGINAL tensors. `train` is the kernel's `train`: "
            "`train.read(spec)` returns what the fwd read with that spec returned on the original head (spec = the same "
            "JSON as the request, as a dict; big arrays come back as numpy arrays), and `train.read(spec, weights=w)` what it "
            "returned on the weights w (a dict of tensors; missing names are the original). It only serves reads you "
            "already ran with the fwd tool, so it costs no pass. recipe.py may use torch, np and math, and may not read "
            "or write files. At final it is run on this run's train data, and it must give back your saved edits "
            "(max abs difference 1e-5); later it is run on other train sets with no agent. So compute everything that "
            "depends on the images (class means, directions, thresholds) from `train` inside edit(); plain constants (a "
            "strength, a scale) can be written in. Test it with `check_recipe()` before final.")
    return notes


def render_prompt(run: Path, project: str, budget: int, max_exec: int) -> str:
    """The message both agents get: the run template, then the task statement.

    `python -m <pkg>.harness CMD` becomes the venv's console script `CMD` (same program). Some agent CLIs will not
    let an allow rule approve a Python interpreter given inline code (`python ... exec -c '...'`),
    and every agent gets the same text so the arms read the same prompt.
    """
    agent_prompt = (run / "AGENT_PROMPT.md").read_text()
    py = harness_python(agent_prompt)
    cmd = f"{py} -m {PKG}.harness"
    if py and Path(py).with_name(CLI).exists():
        cmd = str(Path(py).with_name(CLI))
        agent_prompt = agent_prompt.replace(f"{py} -m {PKG}.harness", cmd)
    return RUN_TEMPLATE.read_text().format(run=run, project=project, budget=budget, max_exec=max_exec, cmd=cmd,
                                           agent_prompt=agent_prompt.strip())


def harness_python(agent_prompt: str) -> str | None:
    """The python the task statement tells the agent to call (the server's own interpreter)."""
    # The reading prompt has no `status` line, only exec/final; missing it falls back to bare
    # `python`, and the agent CLI's allow rules then deny every call.
    m = re.search(rf"`(\S+) -m {re.escape(PKG)}\.harness (?:status|exec|final) ", agent_prompt)
    return m.group(1) if m else None
