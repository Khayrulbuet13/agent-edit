"""Data paths come from AGENT_EDIT_* roots, so the workspace can move without links."""

import json


def test_relative_manifest_source_resolves_against_roots(tmp_path, monkeypatch):
    from agent_edit.data import registry as R

    data, raw = tmp_path / "data", tmp_path / "raw"
    (data / "manifests").mkdir(parents=True)
    monkeypatch.setenv("AGENT_EDIT_DATA", str(data))
    monkeypatch.setenv("AGENT_EDIT_DATA_RAW", str(raw))
    project = R.EXTRA_PROJECTS[0]
    src = {"images": "birds/images", "annotations": "birds/ann.json"}
    (data / "manifests" / f"{project}.json").write_text(json.dumps({"source": src}))
    R.manifest.cache_clear()
    try:
        got = R.manifest(project)["source"]
        assert got == {"images": str(raw / "birds/images"), "annotations": str(data / "birds/ann.json")}

        (data / "manifests" / f"{project}.json").write_text(json.dumps({"source": {"images": "/abs/img"}}))
        R.manifest.cache_clear()
        assert R.manifest(project)["source"]["images"] == "/abs/img"
    finally:
        R.manifest.cache_clear()
