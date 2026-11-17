"""Which projects exist, where their manifests live, and their classes and prompts.

Manifests of the sets built for this package live in $AGENT_EDIT_DATA/manifests. To add a set: put its
manifest there and add its name to EXTRA_PROJECTS (and to PER_CLASS_SHOT if N counts images per
class). Four more sets come from an optional bridge module; they exist only when it ships.
"""

from __future__ import annotations

import importlib
import json
from functools import cache
from pathlib import Path

from .. import config

_BRIDGE = f"{config.PKG}.{config.BRIDGE_FILE}"


def _has_bridge() -> bool:
    return config.has_module(config.BRIDGE_FILE)


def bridge():
    """The optional bridge module. Import errors inside it surface as they are."""
    if not _has_bridge():
        raise RuntimeError("this tree ships no bridge module, which this project or prompt variant needs")
    return importlib.import_module(_BRIDGE)


BRIDGE_PROJECTS = (("livecell-8type", "bbbc038-nuclei", "isaid-buildings", "coco-vehicles")
                   if _has_bridge() else ())
FG_PROJECTS = ("stanford-cars-10", "cub-10", "pets-10", "aircraft-10")
# -mix5: the 10 classes of each FG set randomly paired into 5 groups that do not exist in the world.
MIX5_PROJECTS = tuple(f"{p}-mix5" for p in FG_PROJECTS)
NEW_PROJECTS = ("xview-ships-10",)
# Further manifest-based sets live in an optional module, used when present.
if config.has_module("data.lab_sets"):
    from . import lab_sets as LAB

    CDFSOD_OFFICIAL = LAB.CDFSOD_OFFICIAL
    NEW_PROJECTS = NEW_PROJECTS + LAB.NEW_PROJECTS
EXTRA_PROJECTS = FG_PROJECTS + MIX5_PROJECTS + NEW_PROJECTS
PROJECTS = BRIDGE_PROJECTS + EXTRA_PROJECTS
# N counts images per class here; the train list interleaves classes so every prefix is balanced.
PER_CLASS_SHOT = set(EXTRA_PROJECTS)
N_TRAIN = 20
PROMPT_VARIANT = "literal"


def check_project(project: str) -> None:
    if project not in PROJECTS:
        raise ValueError(f"unknown project {project!r}; expected one of {PROJECTS}")


def manifest_path(project: str) -> Path:
    check_project(project)
    if project in EXTRA_PROJECTS:
        return config.data_dir() / "manifests" / f"{project}.json"
    return bridge().manifest_path(project)


@cache
def manifest(project: str) -> dict:
    """Relative `source` paths resolve against $AGENT_EDIT_DATA_RAW (images) and $AGENT_EDIT_DATA (annotations)."""
    man = json.loads(manifest_path(project).read_text())
    src = man.get("source")
    if isinstance(src, dict):
        roots = {"images": config.data_raw_dir, "annotations": config.data_dir}
        man["source"] = {k: str(roots[k]() / v) if k in roots and not Path(v).is_absolute() else v
                         for k, v in src.items()}
    return man


def class_table(project: str) -> list[dict]:
    return [{"class_id": int(t["id"]), "class_name": str(t["name"])} for t in manifest(project)["taxonomy"]]


def literal_phrase(name: str) -> str:
    """Class name as a bare noun phrase: underscores out, LVIS parentheticals out."""
    return name.split("_(")[0].replace("_", " ").strip()


def class_prompts(project: str, variant: str = PROMPT_VARIANT) -> list[dict]:
    """[{class_id, class_name, prompt}]: the text prompt of each class, for backends that take one.

    'literal' is the bare class name, so a head edit is judged against the stock call. Prompts are
    distinct per class in every project, so one run per prompt carries class identity. The other
    variants need the bridge module.
    """
    if variant == "literal":
        return [{**c, "prompt": literal_phrase(c["class_name"])} for c in class_table(project)]
    probe = bridge().sam3_probe()
    return [{**c, "prompt": probe.phrase_for(project, c["class_name"], variant)} for c in class_table(project)]


def prompt_variants(project: str) -> dict:
    return {v: class_prompts(project, v) for v in bridge().sam3_probe().VARIANTS}
