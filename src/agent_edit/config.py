"""Paths outside the repo. Every one comes from an env var; nothing is hardcoded.

Required (read on first use, so importing the package never needs them):
    AGENT_EDIT_DATA       project-made data: manifests in AGENT_EDIT_DATA/manifests/<project>.json
    AGENT_EDIT_CACHE      frozen-feature caches and model weights
    AGENT_EDIT_RUNS       agent run folders, AGENT_EDIT_RUNS/<project>/<run>
    AGENT_EDIT_RESULTS    scores, baselines, tables

The optional bridge module (data/sources/bridge.py) may add variables of its own; it lists them in
its ENV dict, and `env` shows them only when the module is present.

Needed only by the commands that read raw upstream datasets (COCO for the retention check and the reading test):
    AGENT_EDIT_DATA_RAW   raw upstream datasets, one folder each (coco/annotations, coco/train2017, ...)

Optional, each with a default under the ones above:
    AGENT_EDIT_PMT_ROOT      PMT source, weights, cache   default AGENT_EDIT_CACHE/pmt
    AGENT_EDIT_COCO_ROOT     COCO 2017 (retention, reading) default AGENT_EDIT_DATA_RAW/coco
    AGENT_EDIT_PMT_BACKBONE  DINOv3 HF id or local path   default facebook/dinov3-vitl16-pretrain-lvd1689m
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from importlib.machinery import all_suffixes
from pathlib import Path

PKG = __package__.partition(".")[0]


def cli_name(pkg: str) -> str:
    """The command is the package name with dashes: the console script of `agent_edit` is `agent-edit`."""
    return pkg.replace("_", "-")


CLI = cli_name(PKG)
BRIDGE_FILE = "data.sources.bridge"
_SUFFIXES = all_suffixes()

REQUIRED = {
    "AGENT_EDIT_DATA": "the project-made data folder (holds manifests/<project>.json for the sets built here)",
    "AGENT_EDIT_CACHE": "the frozen-feature cache folder",
    "AGENT_EDIT_RUNS": "the folder of agent runs (<project>/<run>)",
    "AGENT_EDIT_RESULTS": "the folder for scores, baselines and tables",
}
OTHER = {"AGENT_EDIT_DATA_RAW": "the raw upstream dataset folder (coco/, ...)"}


class ConfigError(RuntimeError):
    pass


def bridge_env() -> dict:
    """{var: what it points to} of the bridge module; empty when the module is absent."""
    if not has_module(BRIDGE_FILE):
        return {}
    return getattr(importlib.import_module(f"{PKG}.{BRIDGE_FILE}"), "ENV", {})


def has_module(rel: str) -> bool:
    """Whether this package has the module `rel` (dotted, e.g. 'analysis.replay'), judged
    from the files on disk. Parent packages are not imported (some pull in torch), and a folder
    without an __init__.py, what is left after its files are removed, does not count."""
    spec = importlib.util.find_spec(PKG)
    dirs = [Path(d) for d in (spec.submodule_search_locations or [])] if spec else []
    *parents, leaf = rel.split(".")
    for part in parents:
        dirs = [d / part for d in dirs if (d / part / "__init__.py").is_file()]
    return any((d / leaf / "__init__.py").is_file() or any((d / (leaf + sfx)).is_file() for sfx in _SUFFIXES)
               for d in dirs)


def env_path(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        what = REQUIRED.get(var) or bridge_env().get(var) or OTHER[var]
        raise ConfigError(f"{var} is not set. Point it to {what}. The README lists the variables.")
    return Path(val).expanduser()


def _opt(var: str, default) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default()


def data_dir() -> Path:
    return env_path("AGENT_EDIT_DATA")


def data_raw_dir() -> Path:
    return env_path("AGENT_EDIT_DATA_RAW")


def cache_dir() -> Path:
    return env_path("AGENT_EDIT_CACHE")


def runs_dir() -> Path:
    return env_path("AGENT_EDIT_RUNS")


def results_dir() -> Path:
    return env_path("AGENT_EDIT_RESULTS")


def pmt_root() -> Path:
    return _opt("AGENT_EDIT_PMT_ROOT", lambda: cache_dir() / "pmt")


def coco_root() -> Path:
    return _opt("AGENT_EDIT_COCO_ROOT", lambda: data_raw_dir() / "coco")


def pmt_backbone() -> str:
    return os.environ.get("AGENT_EDIT_PMT_BACKBONE", "facebook/dinov3-vitl16-pretrain-lvd1689m")


def resolve_run(run) -> Path:
    """A run folder given as a path, or as '<project>/<run>' under AGENT_EDIT_RUNS."""
    p = Path(run).expanduser()
    if p.is_dir() or p.is_absolute():
        return p.resolve()
    return (runs_dir() / p).resolve()


def check() -> dict:
    """{var: value or None} for the required vars; the bridge's only when the bridge is present."""
    names = [*REQUIRED, *bridge_env()]
    return {v: os.environ.get(v) for v in names}


if has_module("config_lab"):  # names of an optional sibling module, when the tree has it
    from .config_lab import *  # noqa: F403
