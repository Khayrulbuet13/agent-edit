"""The dataset configs that ship must agree with the constants the code actually uses."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FG = ("stanford-cars-10", "cub-10", "pets-10", "aircraft-10")
SHIPPED = (*FG, *(f"{p}-mix5" for p in FG), "xview-ships-10")
DATASETS = [ROOT / "configs" / "datasets" / f"{p}.yaml" for p in SHIPPED]


def test_every_shipped_set_has_a_dataset_config():
    from agent_edit import data

    assert set(SHIPPED) <= set(data.PROJECTS)
    assert all(p.exists() for p in DATASETS)


@pytest.mark.parametrize("path", DATASETS, ids=lambda p: p.stem)
def test_dataset_config_matches_code(path):
    pytest.importorskip("pycocotools")
    from agent_edit import data
    from agent_edit.backends.pmt import backend as PB

    ds = yaml.safe_load(path.read_text())
    p = ds["name"]
    assert p in data.EXTRA_PROJECTS and ds["manifest"] == "derived"
    assert ds["shot_unit"] == ("image_per_class" if p in data.PER_CLASS_SHOT else "image")
    assert ds["pmt_mask_source"] == PB.SOURCES.get(p, "pmt")
    assert ds["match"] == ("box" if p in PB.BOX_PROJECTS else "mask")
