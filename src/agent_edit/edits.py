"""Reading a finished run: the edited tensors it submitted."""

from __future__ import annotations

import json
from pathlib import Path

from .config import has_module


def load_edit(run_dir) -> dict:
    import torch

    d = Path(run_dir) / "edit"
    man = json.loads((d / "manifest.json").read_text())
    return {n: torch.load(d / v["file"], map_location="cpu", weights_only=True).float() for n, v in man["edited"].items()}


def merge(orig: dict, edit: dict) -> dict:
    return {**orig, **edit}


if has_module("edits_lab"):  # names of an optional sibling module, when the tree has it
    from .edits_lab import *  # noqa: F403
