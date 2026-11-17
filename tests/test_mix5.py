"""The four "-mix5" invented-class projects: each fine-grained set with its 10 classes paired into 5
groups that do not exist in the world.

The manifests are data; this covers the code that reads them: registry membership, the cache alias
(a mix5 set has no cache of its own), and the backend wiring (no species word in its own task text,
secret words for its parent's).
"""

import pytest

from agent_edit import data
from conftest import need_manifest


@pytest.fixture(autouse=True)
def _mix5_manifests():
    need_manifest("cub-10-mix5")


MIX5 = ("stanford-cars-10-mix5", "cub-10-mix5", "pets-10-mix5", "aircraft-10-mix5")
PARENT = dict(zip(MIX5, ("stanford-cars-10", "cub-10", "pets-10", "aircraft-10")))


@pytest.mark.parametrize("project", MIX5)
def test_registered_as_a_per_class_extra_project(project):
    assert project in data.EXTRA_PROJECTS
    assert project in data.PER_CLASS_SHOT
    assert project in data.PROJECTS


@pytest.mark.parametrize("project", MIX5)
def test_manifest_has_5_invented_groups_and_a_cache_alias(project):
    man = data.manifest(project)
    assert man["cache_from"] == PARENT[project]
    names = [t["name"] for t in man["taxonomy"]]
    assert len(names) == 5
    assert all(n.startswith("group_") for n in names)  # no species/breed word


@pytest.mark.parametrize("project", MIX5)
def test_cache_source_root_aliases_the_parent(project):
    pytest.importorskip("pycocotools")
    from agent_edit.backends.pmt import cache as C

    assert C.cache_project(project) == PARENT[project]
    assert C.source_root(project) == C.source_root(PARENT[project])
    assert C.source_root(project).name == PARENT[project]


@pytest.mark.parametrize("project", MIX5)
def test_task_text_has_no_class_or_species_word(project):
    pytest.importorskip("pycocotools")
    from agent_edit.backends.pmt import backend as PB

    task = PB.TASKS[project].lower()
    for word in ("group_1", "group_2", "group_3", "group_4", "group_5"):
        assert word not in task
    parent_names = [t["name"] for t in data.manifest(PARENT[project])["taxonomy"]]
    for name in parent_names:
        for w in name.lower().split():
            assert w not in task.split(), (project, name)


@pytest.mark.parametrize("project", MIX5)
def test_box_projects_match_the_parent_set(project):
    from agent_edit.backends.pmt import backend as PB

    assert (project in PB.BOX_PROJECTS) == (PARENT[project] in PB.BOX_PROJECTS)


@pytest.mark.parametrize("project", MIX5)
def test_fg_parent_row_is_the_base_sets_row(project):
    from agent_edit.backends.pmt import backend as PB

    assert PB.FG_PARENT_COCO_ROW[project] == PB.FG_PARENT_COCO_ROW[PARENT[project]]
    assert PB.FG_PARENT_NAME[project] == PB.FG_PARENT_NAME[PARENT[project]]


@pytest.mark.parametrize("project", MIX5)
def test_secret_words_are_the_parents_real_classes(project, monkeypatch):
    pytest.importorskip("pycocotools")
    from agent_edit.backends.pmt import backend as PB

    monkeypatch.setattr(PB.PmtHeadBackend, "_ensure", lambda self: None)
    be = PB.make_backend(project)
    secret = be.secret_words()
    parent_names = [t["name"] for t in data.manifest(PARENT[project])["taxonomy"]]
    assert sorted(secret) == sorted(parent_names)


def test_base_sets_have_no_secret_words(monkeypatch):
    pytest.importorskip("pycocotools")
    from agent_edit.backends.pmt import backend as PB

    for project in PARENT.values():
        be = PB.make_backend(project)
        assert be.secret_words() == []


def test_image_keys_leak_the_species_until_scrubbed():
    """mix5 keeps the parent's image keys verbatim, so they still spell the parent's real class."""
    man = data.manifest("cub-10-mix5")
    keys = man["train_pool_by_class"]["group_1"]
    assert any("Gull" in k or "Tern" in k for k in keys)
